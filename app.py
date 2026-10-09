"""molabi desktop window: runs the pool server and shows its dashboard."""
import sys
import webview
import pool

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
pool.serve(port)
webview.create_window("molabi", f"http://127.0.0.1:{port}/", width=1280, height=820, min_size=(880, 560),
                      background_color="#121614")  # no white flash before the page paints
webview.start()
