import asyncio
from attrs import fields
import grpc
import uuid
from .message_pb2 import SimpleMessage 
from .message_pb2_grpc import MessagesStub 
from .ChunkWriter import ChunkWriter 
from .ChunkReader import ChunkReader 
import os
from .Models import Chat, Embedding
from .LlmOutput import CachedEntry, RunMetaData, LlmOutput, LlmSimpleOutput
import json
import requests
import hashlib
from pathlib import Path
from dataclasses import asdict

class GrpcConnection:
    """
    Owns ONE shared gRPC channel.
    """
    def __init__(self):
        self.channel = grpc.aio.secure_channel(
                    os.environ["LLM_SERVER_URL"],
                        grpc.ssl_channel_credentials()
                )
        self.stub = MessagesStub(self.channel)

    async def close(self):
        await self.channel.close()



class BidirectionalClient:
    """
    A single streaming message client that:
        - connects automatically when created
        - manages its own streams
        - has its own heartbeat
    """


    def __init__(self, engineType: str, connection : GrpcConnection):
        self.engineType = engineType
        self.conn = connection
        self.stub = connection.stub

        self.guid = uuid.uuid4().hex

        self.is_connected = False

        self.call = None
        self.heartbeat_call = None

        self._msg_future = None
        self._heartbeat_future = None
        self._stream_dead = None      # the exception that killed the call stream, once it has died

        self._heartbeat_task = None

    # ----------------------------------------
    # FACTORY METHOD (async init)
    # ----------------------------------------
    @classmethod
    async def create(cls, engineType: str, connection : GrpcConnection):
        while True:
            try:
                self = cls(engineType, connection)
                await self._connect()
                break
            except Exception as e:
                print(f"Error during connection - retrying...")
                # Free what the failed attempt opened: its two streams stay registered on the SHARED channel otherwise,
                # and after enough failures the channel's stream cap is reached and every later handshake times out
                # (connection storms seen in the PCM week runs, Oct 2026).
                self.is_connected = False
                for call in (self.call, self.heartbeat_call):
                    if call is not None:
                        try:
                            call.cancel()
                        except Exception:
                            pass
                await asyncio.sleep(5)
        return self

    # ----------------------------------------
    # STREAM READERS (FIXED)
    # ----------------------------------------
    async def _read_call_stream(self):
        try:
            async for msg in self.call:
                if self._msg_future and not self._msg_future.done():
                    self._msg_future.set_result(msg)
        except grpc.aio.AioRpcError as e:
            # Connection died, stop reading gracefully
            # This prevents "Task exception was never retrieved"
            # We mark as disconnected so sends will fail and trigger retry logic
            self.is_connected = False
            self._fail_pending(e)
        except Exception as e:
            print(f"Unexpected error in call stream (retrying)")
            self.is_connected = False
            self._fail_pending(e)

    def _fail_pending(self, exc):
        """A reply that will never come: fail the waiting send_receive now (and every later one on this dead stream), so
        SendSurely reconnects at once instead of sitting out its 300 s timeout."""
        self._stream_dead = exc
        if self._msg_future and not self._msg_future.done():
            self._msg_future.set_exception(exc)

    async def _read_heartbeat_stream(self):
        try:
            async for msg in self.heartbeat_call:
                if self._heartbeat_future and not self._heartbeat_future.done():
                    self._heartbeat_future.set_result(msg)
        except grpc.aio.AioRpcError as e:
            # Connection died, stop reading gracefully
            pass 
        except Exception as e:
            print(f"Unexpected error in heartbeat stream (retrying)")

    # ----------------------------------------
    # MAIN CONNECT LOGIC
    # ----------------------------------------
    async def _connect(self):
        if self.is_connected:
            return

        # open two independent streaming RPCs
        self.call = self.stub.BidirectionalMessage()
        self.heartbeat_call = self.stub.BidirectionalMessage()

        # start background readers
        asyncio.create_task(self._read_call_stream())
        asyncio.create_task(self._read_heartbeat_stream())

        # handshake
        hello = SimpleMessage(mtype=self.engineType, payload=self.guid.encode())
        res = await asyncio.wait_for(self.send_receive(hello), timeout=10)        

        self.from_scratch = (res.mtype == "fresh")
        print("session:", "first-time" if self.from_scratch else "reused")

        # start heartbeat
        await self.heartbeat_call.write(
            SimpleMessage(mtype="__heartbeat__", payload=self.guid.encode())
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self.is_connected = True

        writer=ChunkWriter()
        writer.write_str(os.environ["LLM_USER_CODE"])
        await asyncio.wait_for(self.send_receive(SimpleMessage(mtype="__initengine__", payload=writer.close())), timeout=30)


    # ----------------------------------------
    # HEARTBEAT LOOP (FIXED)
    # ----------------------------------------
    async def _heartbeat_loop(self):
        try:
            while True:
                if not self.is_connected:
                    break
                await self.heartbeat_call.write(
                    SimpleMessage(mtype="__heartbeat__", payload=self.guid.encode())
                )
                await asyncio.sleep(10)
        except Exception:
            # If writing fails (connection lost), stop the loop
            self.is_connected = False

    # ----------------------------------------
    # SEND/RECEIVE HELPERS
    # ----------------------------------------
    async def send_receive(self, message):
        if self._stream_dead is not None:
            raise ConnectionError("call stream is dead") from self._stream_dead
        loop = asyncio.get_running_loop()
        self._msg_future = loop.create_future()

        await self.call.write(message)
        return await self._msg_future



    async def close(self):
        """
        Gracefully shuts down the streams.
        """
        self.is_connected = False

        # 1. Stop the heartbeat task first so we don't try to write to a closing stream
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # 2. Gracefully close the Main Call
        if self.call:
            try:
                # done_writing() sends an EOF to the server.
                # The server's loop will exit normally, preventing the error.
                await self.call.done_writing()
            except Exception:
                # If the network is already dead, done_writing might fail.
                # In that case, we fall back to a hard cancel.
                self.call.cancel()

        # 3. Gracefully close the Heartbeat Call
        if self.heartbeat_call:
            try:
                await self.heartbeat_call.done_writing()
            except Exception:
                self.heartbeat_call.cancel()


class LlmClient:

    def __init__(self, connection : GrpcConnection):
        self.connection = connection
        self.client = None # Initialize to None

    async def connect(self):
        # CLEANUP: If a client exists, close it before creating a new one
        if self.client is not None:
            await self.client.close()
            
        self.client = await BidirectionalClient.create("llm",self.connection)
    
    def download_blob(self,sas_url):
        response = requests.get(sas_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.content.decode('utf-8')
    
    def dict_to_dataclass(self,cls, data):
        if not isinstance(data, dict):
            return data
        fields = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in data:
                if hasattr(field_type, '__annotations__'):  # Nested dataclass
                    fields[field_name] = self.dict_to_dataclass(field_type, data[field_name])
                else:
                    fields[field_name] = data[field_name]
        return cls(**fields)

    async def SendSurely(self, message : SimpleMessage, expect_llmoutput: bool):
        while True:
            try:
                response = await asyncio.wait_for(self.client.send_receive(message), timeout=300)
                reader=ChunkReader(response.payload)
                if expect_llmoutput:
                    data_dict = json.loads(reader.read_str())
                    llmOut = self.dict_to_dataclass(LlmOutput, data_dict)
                    simpleOut=LlmSimpleOutput(llmOut.answer,llmOut.error)
                    if llmOut.answerReference!=None:
                        #retrieve answer from blob storage
                        data_dict = json.loads(self.download_blob(llmOut.answerReference))
                        simpleOut.answer = self.dict_to_dataclass(CachedEntry, data_dict)
                    return simpleOut
                else:
                    return None
            except Exception as e:
                print(f"Error while trying to send task to server - retrying...")
                # make a new connection
                # The connect() method now handles closing the old broken client
                await self.connect()
                await asyncio.sleep(5)

    async def Ask(self, chat : Chat, tags : list[str], cache_only : bool = False, retries: int = -1):
        #check local cache
        chatJSON=json.dumps(chat.to_dict(), indent=4)
        hex_hash = hashlib.sha256(chatJSON.encode('utf-8')).hexdigest()
        cachePath=Path(f"{os.environ['LLM_CACHE']}/{hex_hash}")
        if cachePath.exists():
            with open(cachePath, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
                return self.dict_to_dataclass(LlmSimpleOutput, data_dict)
        writer=ChunkWriter()
        writer.write_str(chatJSON)
        writer.write_str(json.dumps(tags))
        if cache_only:
            writer.write_int(1)
        else:
            writer.write_int(0)
        writer.write_int(retries)
        output=await self.SendSurely(SimpleMessage(mtype="ask", payload=writer.close()),True)
        if output.error is None:
            with open(cachePath, 'w', encoding='utf-8') as f:
                json.dump(asdict(output), f, indent=4)
        return output

    async def AskBackground(self, chats : list[Chat], tags : list[str], retries: int = -1):
        writer=ChunkWriter()
        writer.write_str(json.dumps(chats, default=lambda obj: obj.to_dict(), indent=4))
        writer.write_str(json.dumps(tags))
        writer.write_int(retries)
        await self.SendSurely(SimpleMessage(mtype="askmany", payload=writer.close()),False) 

    async def Embed(self, input : Embedding, tags : list[str], cache_only : bool = False, retries: int = -1):
        inputJSON=json.dumps(input.to_dict(), indent=4)
        hex_hash = hashlib.sha256(inputJSON.encode('utf-8')).hexdigest()
        cachePath=Path(f"{os.environ['LLM_CACHE']}/{hex_hash}")
        if cachePath.exists():
            with open(cachePath, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
                return self.dict_to_dataclass(LlmSimpleOutput, data_dict)
        writer=ChunkWriter()
        writer.write_str(json.dumps(input.to_dict(), indent=4))
        writer.write_str(json.dumps(tags))
        if cache_only:
            writer.write_int(1)
        else:
            writer.write_int(0)
        writer.write_int(retries)
        output=await self.SendSurely(SimpleMessage(mtype="embed", payload=writer.close()),True)
        with open(cachePath, 'w', encoding='utf-8') as f:
            json.dump(asdict(output), f, indent=4)
        return output

    async def EmbedBackground(self, inputs : list[Embedding], tags : list[str], retries: int = -1):
        writer=ChunkWriter()
        writer.write_str(json.dumps(inputs, default=lambda obj: obj.to_dict(), indent=4))
        writer.write_str(json.dumps(tags))
        writer.write_int(retries)        
        await self.SendSurely(SimpleMessage(mtype="embedmany", payload=writer.close()),False)

    async def Close(self):
        await self.client.close()

class LlmFactory:
    def __init__(self):        
        self.connection = GrpcConnection()

    async def create_client(self):
        client = LlmClient(self.connection)
        await client.connect()
        return client