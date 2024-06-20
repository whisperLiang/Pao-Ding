from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    is_program_healthy = False

    def do_GET(self):
        if self.is_program_healthy:
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(503)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Service Unavailable')

    def log_message(self, format, *args):
        # 覆写此方法以抑制默认的日志行为
        pass

def healthy_check_run(handler_class=HealthCheckHandler):
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, handler_class)
    print('正在启动用于健康检查的HTTP服务器...')
    httpd.serve_forever()

def set_health_status(status):
    # 假设健康状态存储在服务器实例本身
    HealthCheckHandler.is_program_healthy = status

# def main_program_logic():
#     # 此函数代表您的主要程序逻辑
#     # 为了演示，我们将在设置健康状态为真之前模拟延迟
#     import time
#     time.sleep(5)  # 模拟初始化延迟
#     set_health_status(True)
#     print("执行完毕")

# if __name__ == "__main__":
#     # 在单独的线程中启动HTTP服务器
#     server_thread = threading.Thread(target=healthy_check_run)
#     server_thread.start()

#     # 等待服务器启动（这是一个简化；实际上，你可能需要更健壮的机制）
#     import time
#     time.sleep(1)  # 给服务器一个启动的机会

#     # 主要程序逻辑的开始
#     main_program_logic()

#     # 注意：主线程将继续无限期运行，除非明确停止