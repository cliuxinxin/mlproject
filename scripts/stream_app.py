import streamlit as st
from data_utils import *


nlp = b_load_best_model('bid')


st.title('招标信息提取')

text = st.text_area("输入招标信息", "",800)

if st.button("解析"):
    doc = nlp(text)

    for ent in doc.ents:
        st.write(ent.label_,":",ent.text)