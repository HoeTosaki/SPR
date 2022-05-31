import gensim as gs

def train1():
    sens = [['1','2','3','4'],['2','3','4'],['3','4','5','6']]

    w2v = gs.models.Word2Vec(sentences=sens,vector_size=16,window=3,min_count=0,workers=8,sg=1,epochs=5)

    w2v.save('./w2v.model')
    words = list(w2v.wv.key_to_index.keys())
    print('words:',words)
    print('embs',[w2v.wv[str(word)] for word in words])

def train2():
    sens = [['5','6','7','8'],['8','9','10','11'],['1','2','7'],['1','2','8','9']]
    w2v = gs.models.Word2Vec.load('./w2v.model')
    w2v.build_vocab(sens, keep_raw_vocab=True, trim_rule=None, progress_per=10000, update=True)
    w2v.train(sens,epochs=5,total_examples=len(sens))

    words = list(w2v.wv.key_to_index.keys())
    print('words:', words)
    print('embs', [w2v.wv[str(word)] for word in words])

def train3():
    sens1 = [['1', '2', '3', '4'], ['2', '3', '4'], ['3', '4', '5', '6']]
    sens2 = [['5', '6', '7', '8'], ['8', '9', '10', '11'], ['1', '2', '7'], ['1', '2', '8', '9']]
    w2v = gs.models.Word2Vec(sentences=sens1, vector_size=16, window=3, min_count=0, workers=8, sg=1, epochs=5)
    words = list(w2v.wv.key_to_index.keys())
    print('words:', words)
    for word in words:
        print(f'{word}-{w2v.wv[str(word)]}')
    w2v.build_vocab(sens2, keep_raw_vocab=True, trim_rule=None, progress_per=10000, update=True)
    w2v.train(sens2, epochs=5, total_examples=len(sens2))
    words = list(w2v.wv.key_to_index.keys())
    print('words:', words)
    for word in words:
        print(f'{word}-{w2v.wv[str(word)]}')

def train4():
    sens1 = [['1','2','3','4'],['3','4','5']]
    w2v = gs.models.Word2Vec(sentences=sens1, vector_size=16, window=3, min_count=0, workers=8, sg=1, epochs=5)
    words = list(w2v.wv.key_to_index.keys())
    print('words:', words)
    for word in words:
        print(f'{word}-{w2v.wv[str(word)]}')

if __name__ == '__main__':
    print('hello playground6.')
    # train1()
    # train2()
    # train3()
    train4()
