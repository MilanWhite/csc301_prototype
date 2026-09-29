import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from web_app import app, navigation_url


class NavigationTests(unittest.TestCase):
    def test_address_and_search(self):
        self.assertEqual(navigation_url(' example.com/path '), 'https://example.com/path')
        self.assertEqual(navigation_url('http://localhost:8766/'), 'http://localhost:8766/')
        self.assertEqual(navigation_url('mountain lake'), 'https://www.google.com/search?q=mountain+lake')

    def test_unsafe_or_malformed_urls_are_rejected(self):
        for value in ('', 'javascript:alert(1)', 'file:///C:/Windows', 'data:text/html,hello',
                      'ftp://example.com', 'https://user:password@example.com', 'https://'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                navigation_url(value)

    def test_other_websites_cannot_open_browser_control_socket(self):
        client = TestClient(app)
        with self.assertRaises(WebSocketDisconnect) as caught:
            with client.websocket_connect('/ws', headers={'origin': 'https://unrelated.example'}):
                pass
        self.assertEqual(caught.exception.code, 1008)


if __name__ == '__main__':
    unittest.main()
