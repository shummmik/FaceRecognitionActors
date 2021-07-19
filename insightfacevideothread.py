import cv2
import faiss
import insightface
import numpy as np
import pandas as pd
from PIL import ImageFont
from pypika import Tables
from myfunctions import get_emb, get_embeddings_batches, get_users_batches, offset
from streamtojpg import MjpgServer

# URL = 'http://93.87.72.254:8090/mjpg/video.mjpg'
URL = 'http://127.0.0.1:8081/cam.mjpg'
SIZE = (640, 480)
# size
EMB_IM = 512
topn = 5
users, images, embs = Tables('Users', 'Images', 'Embeddings')

model_detect = insightface.model_zoo.get_model('retinaface_r50_v1')
model_detect.prepare(ctx_id=-1, fix_image_size=SIZE[::-1])
model_recog = insightface.model_zoo.get_model('arcface_r100_v1')
model_recog.prepare(ctx_id=-1)

color = (0, 255, 0)
font = cv2.FONT_HERSHEY_COMPLEX

embs_columns = ['id', 'id_user', 'embs']
users_columns = ['id', 'first_name', 'last_name']
embs_frame = pd.DataFrame()
users_frame = pd.DataFrame()

for entries in get_embeddings_batches():
    embs_frame = embs_frame.append(entries, ignore_index=True)

embs_frame.columns = embs_columns

for users in get_users_batches():
    users_frame = users_frame.append(users, ignore_index=True)

users_frame.columns = users_columns

embs_array = np.array(embs_frame['embs'].tolist()).astype(np.float32)

faiss.normalize_L2(embs_array)

index = faiss.index_factory(EMB_IM, "Flat", faiss.METRIC_INNER_PRODUCT)

index.add(embs_array)

mjpgserver = MjpgServer(URL)

fontt = ImageFont.load_default().font

while True:
    frame = mjpgserver.get_frame

    bbox, _ = model_detect.detect(frame)
    for box in bbox.tolist():

        if box[4] > .60:
            delta1 = int(0.1 * (box[2] - box[0]))
            delta2 = int(0.1 * (box[3] - box[1]))

            x1, x2 = offset(int(box[0]) - delta1, int(box[2]) + delta1, SIZE[0])
            y1, y2 = offset(int(box[1]) - delta2, int(box[3]) + delta2, SIZE[1])

            face = frame[y1:y2, x1:x2]

            # resize small and big!!!!!!!!!!!!!!!!
            # face = cv2.resize(face,(112,112), interpolation=cv2.INTER_LINEAR)
            # emb = model_recog.get_embedding(face)

            emb = get_emb(face, model_recog)
            faiss.normalize_L2(emb)
            D, I = index.search(emb, topn)

            if len(I) > 0:
                id_user = embs_frame['id_user'][I[0][0]]
                name_user = ' '.join(users_frame[users_frame['id'] == id_user].values[0][1:])
                frame = cv2.putText(frame, str(name_user), (x2, y2), font, 0.3, color, 1, cv2.LINE_AA)
                frame = cv2.putText(frame, str(D[0][0]), (x2, y2 - 10), font, 0.3, color, 1, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.imshow('Camera', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
