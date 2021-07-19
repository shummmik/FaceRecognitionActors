import numpy as np
from pypika import Tables

users, images, embs = Tables('Users', 'Images', 'Embeddings')


class Entry:
    EMBEDDING_SIZE = 512

    def __init__(self, id_emb, id_user, embedding):
        self.id = np.int64(id_emb)
        self.id_user = id_user
        self.embedding = np.array(embedding, dtype=np.float32)
        self._check_embedding_size()

    def _check_embedding_size(self):
        if self.embedding.shape != (self.EMBEDDING_SIZE,):
            raise ValueError('Invalid embedding size for entry: {0}'.format(self.id))

    @classmethod
    def from_tuple(cls, entry_tuple):
        return cls(
            id_emb=entry_tuple[0],
            id_user=entry_tuple[1],
            embedding=entry_tuple[2]
        )

    def __repr__(self):
        return '{0}(id={1!r}, id_user={2!r}, embedding={3!r})'.format(
            self.__class__.__name__,
            self.id,
            self.id_user,
            self.embedding,
        )
