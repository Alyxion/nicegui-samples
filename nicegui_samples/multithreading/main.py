import time
import base64
from pathlib import Path
import cv2
import os
from nicegui import ui
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../data'

thread_pool = ThreadPoolExecutor(max_workers=16)

# warm up the thread pool
for _ in range(16):
    thread_pool.submit(lambda: time.sleep(0.1))

def compress_thread(img):
    return cv2.imencode('.jpg', img)[1]

total_images = 512

def benchmark(img):
    # compress locally, single threaded
    results = {}
    start = time.time()
    for i in range(total_images):
        results[i] = compress_thread(img)
    end = time.time()
    results['local'] = end - start

    # compress with thread pool
    start = time.time()
    images = list(thread_pool.map(compress_thread, [img] * total_images))  # generator to list conversion actually awaits the threads
    end = time.time()
    results['thread_pool'] = end - start

    ui.notify(f'local: {results["local"]:.2f} seconds')
    ui.notify(f'thread_pool: {results["thread_pool"]:.2f} seconds')

@ui.page("/")
def index():
    # Read the image using OpenCV
    image_path = DATA_DIR + '/dt_green_hills.jpg'
    img = cv2.imread(str(image_path))
    if img is None:
        ui.notify('Failed to load image!', type='negative')
        return

    # base encode image
    image_data = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')
    ui.html(f'<img src="data:image/jpeg;base64,{image_data}" width="{img.shape[1]//2}" height="{img.shape[0]//2}">')

    ui.button('Benchmark', on_click=lambda: benchmark(img))

    

ui.run(title='Image Viewer')
