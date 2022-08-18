import streamlit as st
from data_utils import *

st.title('招标信息提取')

text = st.text_area("输入招标信息", "",800)

# 解析按钮放在最右边
if st.button("解析"):
    nlp = b_load_best_model('tender')

    doc = nlp(text)

    for ent in doc.ents:
        st.write(ent.label_,":",ent.text)


