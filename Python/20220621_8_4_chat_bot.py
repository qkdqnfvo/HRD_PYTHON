from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_chat import message

import pandas as pd
import streamlit as st
import json
# pip install streamlit
# pip install streamlit_chat
# pip install -U sentence-transformers



@st.cache(allow_output_mutation=True)
def cached_model():        
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('./data/chat_bot/dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

if __name__ == '__main__':
    df = get_dataset()
    model = cached_model()
    st.header('심리상담 챗봇')
    st.markdown('[참고사이트](https://naver.com)')

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('>>', '')
        submitted = st.form_submit_button('submit')

    if submitted and user_input:
        em = model.encode(user_input)
        df['cosine'] = df['embedding'].map(lambda x: cosine_similarity([em], [x]))
        answer = df.loc[df['cosine'].values.reshape(-1, 1).argmax()]
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer['챗봇'])
        # st.text_area(st.session_state.generated[-1])

    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i:
            message(st.session_state['generated'][i], key=str(i) + '_bot')