import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

mecab = Mecab()

total_data = pd.read_table('output.csv', sep=',')
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력
total_data.drop(['place', 'time', 'area'], axis='columns', inplace=True)
remove_num = total_data[total_data['rating'] == 3].index
total_data.drop(remove_num, inplace=True)
total_data['label'] = np.select([total_data.rating > 3], [1], default=0)
# print(total_data['area1'].nunique(), total_data['rating'].nunique(), total_data['snippet'].nunique(), total_data['label'].nunique())
total_data.drop_duplicates(subset=['snippet'], inplace=True) # reviews 열에서 중복인 내용이 있다면 중복 제거
print('총 샘플의 수 :',len(total_data))
Busan = total_data.groupby('area1').get_group('부산')
Jeju = total_data.groupby('area1').get_group('제주도')
Seoul = total_data.groupby('area1').get_group('서울')
# print(len(Busan), len(Jeju), len(Seoul)) # 938 944 974

area_list = {'busan': Busan, 'jeju': Jeju, 'seoul': Seoul}

for i in area_list:
    size = int(area_list[i].groupby('label').size()[0])
    # print(area_list['busan'][area_list['busan'].label == 1])
    area_list[i][area_list[i].label == 1] = area_list[i].groupby('label').get_group(1).sample(n=random.randrange(size-15, size+30))
    # area_list['busan']. = area_list['busan'].groupby('label').get_group(1).sample(n=random.randrange(size-10, size+51))
    area_list[i] = area_list[i].dropna(axis=0)

for i in area_list:
    print(f"{i}의 {area_list[i].groupby('label').size().reset_index(name = 'count')}")

for i in area_list:
    area_list[i]['snippet'] = area_list[i]['snippet'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    area_list[i]['snippet'].replace('', np.nan, inplace=True)
    area_list[i] = train_test_split(area_list[i], test_size = 0.25, random_state = 42) # 0 train, 1 test
    area_list[i][1] = area_list[i][1].dropna(how='any')
#print(area_list['jeju'][0])
# print(len(area_list['busan'][0]))
# 한글과 공백을 제외하고 모두 제거


stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯',
             '과', '와', '네', '들', '듯', '지', '임', '게', '있', '습니다', '되', '수', '입니다', '으로',
             '로', '할', '어', '보', '면', '기', '곳', '에서', '것', '지만', '서울', '제주', '부산', '제주',
             '제주도', '합니다', '었', '어요', '어서', '적', '았', '갈', '만', '잘', '나', '않', '번', '아요',
             '는데', '했', '네요', '서', '길', '는데', '아', '없', '같', '해', '때', '까지', '시','사람', 
             '너무', '볼', '방문', '시간', '절', '타', '좋', '많', '러', '들어가', '음', '아서', '아니', '함', '원',
             '생각', '분', '해서', '라', '겠', '안', '구', '싶', '구요', '거', '그냥', '던', '중', '사', '에게',
             '걸', '다는', '님', '싶', '곳곳', '년', '정도', '찍', '긴', '라는', '며', '따라', '추천', '힘들'
             '으면', '줄', '일', '그리고', '부터', '한다', '자', '보다', '에게', '근처', '우리', '정말', '별로',
             '몇', '데', '은데', '진', '갔', '그', '꼭', '니', '올라가', '아주', '면서', '최', '곱', '맑', '날', '니',
             ]

for i in area_list:
    area_list[i][0]['tokenized'] = area_list[i][0]['snippet'].apply(mecab.morphs)
    area_list[i][0]['tokenized'] = area_list[i][0]['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
    area_list[i][1]['tokenized'] = area_list[i][1]['snippet'].apply(mecab.morphs)
    area_list[i][1]['tokenized'] = area_list[i][1]['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

negative_words = {i: np.hstack(area_list[i][0][area_list[i][0].label == 0]['tokenized'].values) for i in area_list}
positive_words = {i: np.hstack(area_list[i][0][area_list[i][0].label == 1]['tokenized'].values) for i in area_list}

negative_word_count = {i: Counter(negative_words[i]) for i in area_list}
positive_word_count = {i: Counter(positive_words[i]) for i in area_list}

print(negative_word_count['busan'].most_common(20))
print(positive_word_count['busan'].most_common(20))


# print(positive_word_count['jeju'].most_common(20))
# print(negative_word_count['jeju'].most_common(20))

X_train = {i: area_list[i][0]['tokenized'].values for i in area_list}
y_train = {i: area_list[i][0]['label'].values for i in area_list}
X_test= {i: area_list[i][1]['tokenized'].values for i in area_list}
y_test = {i: area_list[i][1]['label'].values for i in area_list}

tokenizer = Tokenizer()
for i in X_train:
    tokenizer.fit_on_texts(X_train[i])
# print(tokenizer.word_index)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
for i in X_train:
    tokenizer.fit_on_texts(X_train[i])
X_train = {i: tokenizer.texts_to_sequences(X_train[i]) for i in X_train}
X_test = {i: tokenizer.texts_to_sequences(X_test[i]) for i in X_test}

for i in area_list:
    print(i+'의 리뷰 최대 길이 :',max(len(l) for l in X_train[i]))
    print(i+'의 리뷰 평균 길이 :',sum(map(len, X_train[i]))/len(X_train[i]))

for i in area_list:
    X_train[i] = pad_sequences(X_train[i], maxlen = 80)
    X_test[i] = pad_sequences(X_test[i], maxlen = 80)

from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
for i in area_list:
    model.fit(X_train[i], y_train[i], epochs=20, callbacks=[es, ModelCheckpoint('best_model_'+i+'.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)], batch_size=60, validation_split=0.2)

loaded_model = load_model('./best_model_busan.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test['jeju'], y_test['jeju'])[1]))
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test['busan'], y_test['busan'])[1]))
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test['seoul'], y_test['seoul'])[1]))


for i in area_list:
    print(f'============={i}==============')
    print('naga:',negative_word_count[i].most_common(20))
    print('posi:',positive_word_count[i].most_common(20))
