:: 先进入父目录再执行protoc，这样生成的代码就会从父目录开始import
cd ..
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. rpc/msg.proto