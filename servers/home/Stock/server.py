from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Hello from the Python server!')

        elif self.path == '/data':
            data = {'message': 'Hello from the Python server!'}
            response = json.dumps(data).encode()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response)

        else:
            self.send_response(404)
            self.end_headers()

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleServer)
    print(f"Serving at http://127.0.0.1:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
