from classes import entry as entry_p
import cv2
import numpy as np
from pypika import Query, Tables, Field, Parameter
import psycopg2

users_table, images, embs = Tables('Users', 'Images', 'Embeddings')


def chunked(input_list: list, batch: int = 10):
    for i in range(0, len(input_list), batch):
        yield input_list[i:i + batch]


#   question new rows in database
def get_new_entries_batches_(batch_size: int = 10, limit: int = 1000):
    max_id = 0

    with psycopg2.connect(host='localhost',
                          database='test',
                          user='postgres',
                          password='passqaz1') as conn:
        with conn.cursor() as cursor:
            while True:
                query_select = Query.from_(embs). \
                    select('*'). \
                    orderby('id'). \
                    limit(limit). \
                    where(embs.id > max_id)

                cursor.execute(str(query_select))
                rows = cursor.fetchall()
                if not rows:
                    break

                for batch in chunked(rows, batch_size):
                    entries = []
                    for row in batch:
                        entries.append(entry_p.Entry.from_tuple(row))
                        max_id = max(row[0], max_id)

                    yield entries

                #   question new rows in database


def get_embeddings_batches(batch_size: int = 30, limit: int = 1000):
    max_id = 0

    with psycopg2.connect(host='localhost',
                          database='test',
                          user='postgres',
                          password='passqaz1') as conn:
        with conn.cursor() as cursor:
            while True:
                query_select = Query.from_(embs). \
                    select('*'). \
                    orderby('id'). \
                    limit(limit). \
                    where(embs.id > max_id)

                cursor.execute(str(query_select))
                rows = cursor.fetchall()
                if not rows:
                    break

                for batch in chunked(rows, batch_size):
                    entries = []
                    for row in batch:
                        entries.append([row[0], row[1], np.array(row[2], dtype=np.float32)])
                        max_id = max(row[0], max_id)

                    yield entries


def get_users_batches(batch_size: int = 30, limit: int = 1000):
    max_id = 0
    with psycopg2.connect(host='localhost',
                          database='test',
                          user='postgres',
                          password='passqaz1') as conn:
        with conn.cursor() as cursor:
            while True:
                query_select = Query.from_(users_table). \
                    select('*'). \
                    orderby('id'). \
                    limit(limit). \
                    where(users_table.id > max_id)
                cursor.execute(str(query_select))
                rows = cursor.fetchall()
                if not rows:
                    break
                for batch in chunked(rows, batch_size):
                    entries = []
                    for row in batch:
                        entries.append(row)
                        max_id = max(row[0], max_id)

                    yield entries


def offset(n1, n2, size):
    if n1 < 0:
        n1 = 0
    if n2 > size:
        n2 = size
    return n1, n2


def gray2rgb(array: np.ndarray):
    arr = np.zeros([array.shape[0], array.shape[1], 3])
    arr[:, :, 0] = array
    arr[:, :, 1] = array
    arr[:, :, 2] = array
    return arr


def get_emb(frame, model_recognition):
    if len(frame.shape) == 2:
        frame = gray2rgb(frame)
    else:
        frame = frame[:, :, ::-1]

    max_size = max(frame.shape)
    black = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    black[int((max_size - frame.shape[0]) / 2):
          int((max_size + frame.shape[0]) / 2),
    int((max_size - frame.shape[1]) / 2):
    int((max_size + frame.shape[1]) / 2)] = frame
    face = cv2.resize(black, (112, 112), interpolation=cv2.INTER_LINEAR)
    return model_recognition.get_embedding(face)
