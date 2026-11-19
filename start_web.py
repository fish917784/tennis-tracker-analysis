"""Start the local tennis analysis service."""
import argparse
import uvicorn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8018)
    args = parser.parse_args()
    print(f'Tennis Tracker Web: http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run('working.web.app:app', host='127.0.0.1', port=args.port)
