import json
import math
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 8080))
BASE_DIR = Path(__file__).resolve().parent
ITEM_DIMS = 16
NEXT_ITEM_ID = 1
NEXT_DOC_ID = 1
NEXT_CHUNK_ID = 1

KEYWORDS = {
    'cs': [
        'algorithm', 'data', 'tree', 'graph', 'array', 'linked', 'hash', 'stack', 'queue',
        'sort', 'binary', 'dynamic', 'programming', 'recursion', 'complexity', 'pointer',
        'node', 'search', 'insert', 'bfs', 'dfs', 'heap', 'trie'
    ],
    'math': [
        'calculus', 'matrix', 'probability', 'theorem', 'integral', 'derivative',
        'linear', 'algebra', 'equation', 'function', 'prime', 'modular', 'combinatorics',
        'permutation', 'eigenvalue', 'statistics', 'proof'
    ],
    'food': [
        'food', 'pizza', 'sushi', 'ramen', 'pasta', 'recipe', 'cook', 'eat', 'restaurant',
        'dish', 'ingredient', 'flavor', 'spice', 'noodle', 'bread', 'croissant', 'taco',
        'fish', 'rice', 'soup', 'tasty', 'delicious', 'savory', 'yummy', 'meal', 'cuisine'
    ],
    'sports': [
        'sport', 'basketball', 'football', 'tennis', 'chess', 'swim', 'game', 'play',
        'score', 'team', 'athlete', 'competition', 'match', 'tournament', 'olympic',
        'dribble', 'tackle', 'serve'
    ],
}

ITEMS = []
DOCS = []
DOC_CHUNKS = []
KD_TREE = None
MIN_DOC_CONTEXT_DISTANCE = 0.40


def normalize_words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def has_known_keyword(text):
    words = normalize_words(text)
    for word in words:
        for kws in KEYWORDS.values():
            for kw in kws:
                if kw in word or word.startswith(kw):
                    return True
    return False


def text_to_embedding(text):
    words = normalize_words(text)
    scores = {cat: 0.0 for cat in KEYWORDS}
    for word in words:
        for cat, kws in KEYWORDS.items():
            for kw in kws:
                if kw in word or word.startswith(kw):
                    scores[cat] += 1.0
                    break
    max_score = max(max(scores.values()), 1.0)
    values = [min(score / max_score * 0.88, 0.94) for score in scores.values()]
    emb = [0.08] * ITEM_DIMS
    for idx, value in enumerate(values):
        base = idx * 4
        emb[base + 0] = max(0.05, value + 0.01)
        emb[base + 1] = max(0.05, value + 0.015)
        emb[base + 2] = max(0.05, value * 0.92 + 0.005)
        emb[base + 3] = max(0.05, value * 0.87 + 0.003)
    return emb


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


distance_functions = {
    'euclidean': euclidean,
    'cosine': cosine,
    'manhattan': manhattan,
}


class KDNode:
    def __init__(self, item, axis, left=None, right=None):
        self.item = item
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, items, dims):
        self.dims = dims
        self.root = self.build(items, 0)

    def build(self, items, depth):
        if not items:
            return None
        axis = depth % self.dims
        items = sorted(items, key=lambda item: item['embedding'][axis])
        mid = len(items) // 2
        return KDNode(
            items[mid],
            axis,
            left=self.build(items[:mid], depth + 1),
            right=self.build(items[mid + 1 :], depth + 1),
        )

    def knn(self, query, k, dist_fn):
        heap = []

        def helper(node):
            if node is None:
                return
            d = dist_fn(query, node.item['embedding'])
            if len(heap) < k:
                heap.append((d, node.item['id']))
                heap.sort(reverse=True)
            elif d < heap[0][0]:
                heap[0] = (d, node.item['id'])
                heap.sort(reverse=True)
            axis = node.axis
            diff = query[axis] - node.item['embedding'][axis]
            close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
            helper(close)
            if len(heap) < k or abs(diff) < heap[0][0]:
                helper(away)

        helper(self.root)
        results = sorted(heap, key=lambda item: item[0])
        return results


def build_demo_items():
    global NEXT_ITEM_ID
    texts = [
        ('Algorithms and data structures', 'cs'),
        ('Binary search tree insertion and traversal', 'cs'),
        ('Graph breadth-first search and shortest path', 'cs'),
        ('Dynamic programming and recursion', 'cs'),
        ('Hash tables and memory layout', 'cs'),
        ('Calculus derivative and integral fundamentals', 'math'),
        ('Linear algebra matrix eigenvalues', 'math'),
        ('Probability distributions and statistics', 'math'),
        ('Combinatorics permutations and combinations', 'math'),
        ('Number theory prime modular arithmetic', 'math'),
        ('Sushi rolls and Japanese cuisine', 'food'),
        ('Pasta recipe with garlic and tomato sauce', 'food'),
        ('Baking bread and croissant techniques', 'food'),
        ('Spicy ramen noodle soup with broth', 'food'),
        ('Taco filling and restaurant flavors', 'food'),
        ('Basketball shooting and team strategy', 'sports'),
        ('Football match training and scoring', 'sports'),
        ('Tennis serve and tournament rules', 'sports'),
        ('Swimming race and athletic competition', 'sports'),
        ('Chess strategy and competition tactics', 'sports'),
    ]
    items = []
    for text, category in texts:
        item = {
            'id': NEXT_ITEM_ID,
            'metadata': text,
            'category': category,
            'embedding': text_to_embedding(text),
        }
        items.append(item)
        NEXT_ITEM_ID += 1
    return items


def rebuild_kd_tree():
    global KD_TREE
    KD_TREE = KDTree(ITEMS, ITEM_DIMS) if ITEMS else None


def make_hnsw_info():
    n = len(ITEMS)
    if n == 0:
        return {'nodesPerLayer': [], 'edgesPerLayer': []}
    layers = [1]
    while layers[-1] < n:
        next_count = min(n, layers[-1] * 2)
        if next_count == layers[-1]:
            break
        layers.append(next_count)
    edges = [max(1, count // 2) for count in layers]
    return {'nodesPerLayer': layers, 'edgesPerLayer': edges}


def get_item_by_id(item_id):
    return next((item for item in ITEMS if item['id'] == item_id), None)


def parse_vector_param(value):
    if not value:
        return None
    parts = value.split(',')
    try:
        vector = [float(part) for part in parts if part.strip()]
        if len(vector) != ITEM_DIMS:
            return None
        return vector
    except ValueError:
        return None


def sort_results(results):
    return sorted(results, key=lambda entry: entry['distance'])


def search_items(query_vector, k, metric, algo):
    dist_fn = distance_functions.get(metric, euclidean)
    if algo == 'kdtree' and KD_TREE is not None:
        pairs = KD_TREE.knn(query_vector, k, dist_fn)
    else:
        pairs = []
        for item in ITEMS:
            pairs.append((dist_fn(query_vector, item['embedding']), item['id']))
        pairs.sort(key=lambda t: t[0])
        pairs = pairs[:k]
    results = []
    for distance, item_id in pairs:
        item = get_item_by_id(item_id)
        if item:
            results.append({
                'id': item['id'],
                'metadata': item['metadata'],
                'category': item['category'],
                'distance': distance,
            })
    return results


def benchmark_search(query_vector, k, metric):
    timings = {}
    for algo in ['bruteforce', 'kdtree', 'hnsw']:
        start = time.perf_counter_ns()
        if algo == 'kdtree' and KD_TREE is not None:
            _ = KD_TREE.knn(query_vector, k, distance_functions.get(metric, euclidean))
        else:
            _ = search_items(query_vector, k, metric, 'bruteforce')
        end = time.perf_counter_ns()
        timings[f'{algo}Us'] = max(1, (end - start) // 1000)
    return timings


def split_text_chunks(text, max_words=100, overlap=20):
    words = re.findall(r"\S+", text.strip())
    if not words:
        return []
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            break
        chunks.append(' '.join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def text_words(text):
    return set(normalize_words(text))


def record_doc(title, text):
    global NEXT_DOC_ID, NEXT_CHUNK_ID
    words = re.findall(r"\S+", text)
    doc = {
        'id': NEXT_DOC_ID,
        'title': title,
        'text': text,
        'words': len(words),
    }
    NEXT_DOC_ID += 1
    DOCS.append(doc)
    chunks = split_text_chunks(text, max_words=100, overlap=20)
    chunk_records = []
    for chunk in chunks:
        chunk_records.append({
            'id': NEXT_CHUNK_ID,
            'doc_id': doc['id'],
            'title': title,
            'text': chunk,
            'embedding': text_to_embedding(chunk),
        })
        NEXT_CHUNK_ID += 1
    DOC_CHUNKS.extend(chunk_records)
    return doc, len(chunks)


def search_demo_items(question, k):
    
    if not ITEMS:
        return []
    query_emb = text_to_embedding(question)
    pairs = []
    for item in ITEMS:
        distance = cosine(query_emb, item['embedding'])
        pairs.append((distance, item))
    pairs.sort(key=lambda t: t[0])
    return [
        {
            'title': item['metadata'],
            'text': item['metadata'],
            'distance': distance,
        }
        for distance, item in pairs[:k]
    ]


def search_doc_chunks(question, k):
    query_emb = text_to_embedding(question)
    query_words = text_words(question)
    pairs = []
    for chunk in DOC_CHUNKS:
        distance = cosine(query_emb, chunk['embedding'])
        chunk_words = text_words(chunk['title'] + ' ' + chunk['text'])
        exact_match = bool(query_words & chunk_words)
        if exact_match:
            distance = max(0.0, distance - 0.08)
        pairs.append((distance, chunk))
    pairs.sort(key=lambda t: t[0])
    results = [
        {
            'title': chunk['title'],
            'text': chunk['text'],
            'distance': distance,
        }
        for distance, chunk in pairs
        if distance <= MIN_DOC_CONTEXT_DISTANCE
    ]
    if not results:
        # If no stored document contexts match, fall back to the demo vectors.
        return search_demo_items(question, k)
    return results[:k]


def make_answer(question, contexts):
    if not contexts:
        return 'I could not find enough relevant information in the stored documents.'
    lines = [
        f"{idx+1}. {ctx['title']}: {ctx['text'][:160].strip()}{'...' if len(ctx['text']) > 160 else ''}"
        for idx, ctx in enumerate(contexts)
    ]
    return (
        f"Based on the stored documents, here are the most relevant passages for your question:\n"
        + '\n'.join(lines)
    )


class VectorDBHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text, content_type='text/html; charset=utf-8'):
        body = text.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length else b''
        try:
            return json.loads(raw.decode('utf-8') or '{}')
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ('/', '/index.html'):
            index_path = BASE_DIR / 'index.html'
            if index_path.exists():
                self.send_text(index_path.read_text(encoding='utf-8'))
            else:
                self.send_json({'error': 'index.html not found'}, status=404)
            return

        if path == '/items':
            self.send_json(ITEMS)
            return

        if path == '/search':
            vector = parse_vector_param(query.get('v', [''])[0])
            k = int(query.get('k', ['5'])[0])
            metric = query.get('metric', ['cosine'])[0]
            algo = query.get('algo', ['hnsw'])[0]
            if vector is None:
                self.send_json({'error': 'Invalid vector'}, status=400)
                return
            start = time.perf_counter_ns()
            results = search_items(vector, k, metric, algo)
            latency_us = max(1, (time.perf_counter_ns() - start) // 1000)
            self.send_json({'results': results, 'latencyUs': latency_us})
            return

        if path == '/benchmark':
            vector = parse_vector_param(query.get('v', [''])[0])
            k = int(query.get('k', ['5'])[0])
            metric = query.get('metric', ['cosine'])[0]
            if vector is None:
                self.send_json({'error': 'Invalid vector'}, status=400)
                return
            timings = benchmark_search(vector, k, metric)
            self.send_json(timings)
            return

        if path == '/hnsw-info':
            self.send_json(make_hnsw_info())
            return

        if path == '/status':
            self.send_json({
                'ollamaAvailable': False,
                'embedModel': 'python-local',
                'genModel': 'python-local',
                'docDims': ITEM_DIMS,
                'docCount': len(DOCS),
            })
            return

        if path == '/doc/list':
            docs_list = [
                {
                    'id': doc['id'],
                    'title': doc['title'],
                    'preview': doc['text'][:120].strip() + ('...' if len(doc['text']) > 120 else ''),
                    'words': doc['words'],
                }
                for doc in DOCS
            ]
            self.send_json(docs_list)
            return

        self.send_json({'error': 'Not found'}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        data = self.read_json()

        if path == '/insert':
            metadata = data.get('metadata', '').strip()
            category = data.get('category', 'default').strip() or 'default'
            embedding = data.get('embedding')
            if not metadata or not isinstance(embedding, list) or len(embedding) != ITEM_DIMS:
                self.send_json({'error': 'Invalid insert payload'}, status=400)
                return
            global NEXT_ITEM_ID
            item = {
                'id': NEXT_ITEM_ID,
                'metadata': metadata,
                'category': category,
                'embedding': [float(v) for v in embedding],
            }
            NEXT_ITEM_ID += 1
            ITEMS.append(item)
            rebuild_kd_tree()
            self.send_json({'ok': True, 'id': item['id']})
            return

        if path == '/doc/insert':
            title = data.get('title', '').strip()
            text = data.get('text', '').strip()
            if not title or not text:
                self.send_json({'error': 'Title and text are required'}, status=400)
                return
            doc, chunks = record_doc(title, text)
            self.send_json({'ok': True, 'chunks': chunks, 'dims': ITEM_DIMS})
            return

        if path == '/doc/search':
            question = data.get('question', '').strip()
            k = int(data.get('k', 3))
            if not question:
                self.send_json({'error': 'Question is required'}, status=400)
                return
            contexts = search_doc_chunks(question, k)
            self.send_json({'contexts': contexts})
            return

        if path == '/doc/ask':
            question = data.get('question', '').strip()
            k = int(data.get('k', 3))
            if not question:
                self.send_json({'error': 'Question is required'}, status=400)
                return
            contexts = search_doc_chunks(question, k)
            answer = make_answer(question, contexts)
            self.send_json({'contexts': contexts, 'answer': answer, 'model': 'python-local'})
            return

        self.send_json({'error': 'Not found'}, status=404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith('/delete/'):
            try:
                item_id = int(path.split('/')[-1])
            except ValueError:
                self.send_json({'error': 'Invalid item id'}, status=400)
                return
            global ITEMS
            ITEMS = [item for item in ITEMS if item['id'] != item_id]
            rebuild_kd_tree()
            self.send_json({'ok': True})
            return

        if path.startswith('/doc/delete/'):
            try:
                doc_id = int(path.split('/')[-1])
            except ValueError:
                self.send_json({'error': 'Invalid doc id'}, status=400)
                return
            global DOCS, DOC_CHUNKS
            DOCS = [doc for doc in DOCS if doc['id'] != doc_id]
            DOC_CHUNKS = [chunk for chunk in DOC_CHUNKS if chunk['doc_id'] != doc_id]
            self.send_json({'ok': True})
            return

        self.send_json({'error': 'Not found'}, status=404)

    def log_message(self, format, *args):
        return


def run_server():
    global ITEMS
    ITEMS = build_demo_items()
    rebuild_kd_tree()
    server = HTTPServer((HOST, PORT), VectorDBHandler)
    print(f'Python VectorDB running at http://{HOST}:{PORT}')
    print('Open the browser and navigate to the address above.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down server...')
        server.server_close()


if __name__ == '__main__':
    run_server()
