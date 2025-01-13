#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Hello from GET request!")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        # Process post_data here

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Received POST request!")

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHandler)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()