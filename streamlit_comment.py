import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import regex
import string
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from underthesea import word_tokenize, pos_tag, sent_tokenize

from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import time

import pandas as pd
import re

# 1. Load data
san_pham = pd.read_csv('San_pham.csv')
data = pd.read_csv('Danh_gia.csv', encoding='utf-8')
X_train = pd.read_csv('x_train.csv', index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0)
X_test = pd.read_csv('x_test.csv', index_col=0)
y_test = pd.read_csv('y_test.csv', index_col=0)
pre_df = pd.read_csv('prepared_4_data.csv', encoding='utf-8')
data_visual = pre_df[['joined_positive_words', 'joined_negative_words']].fillna('').astype(str)

merged_data = pd.merge(data, san_pham[['ma_san_pham', 'ten_san_pham']], on='ma_san_pham', how='left')

# 2. hàm làm sạch dữ liệu
## Đọc file icon và words
with open('negative_icon.txt', 'r', encoding='utf-8') as file:
    icon_nev = [icon.strip() for icon in file.read().splitlines() if icon.strip()]

#file icon tích cực
with open('positive_icon.txt', 'r', encoding='utf-8') as file:
    icon_pov = [icon.strip() for icon in file.read().splitlines() if icon.strip()]

#file word tiêu cực
with open('negative_word.txt', 'r', encoding='utf-8') as file:
    word_nev = [word.strip() for word in file.read().split('\n') if word.strip()]

#file word tích cực
with open('positive_word.txt', 'r', encoding='utf-8') as file:
    word_pov = [word.strip() for word in file.read().split('\n') if word.strip()]

#file từ điển tiếng Việt
with open('vie_dict.txt', 'r', encoding='utf-8') as file:
    vie_dict = [word.strip() for word in file.read().split('\n') if word.strip()]

#file emoji chuyển sang cảm xúc bằng lời
with open('emojicon_copy.txt', 'r', encoding="utf8") as file:
    emoji_dict = {line.split('\t')[0]: line.split('\t')[1].strip() for line in file if '\t' in line}

#file teen code
with open('teencode.txt', 'r', encoding='utf8') as file:
    teen_dict = {line.split('\t')[0]: line.split('\t')[1].strip() for line in file if '\t' in line}

#file từ điển Anh-Việt
with open('english-vnmese.txt', 'r', encoding='utf8') as file:
    english_dict= {line.split('\t')[0]: line.split('\t')[1].strip() for line in file if '\t' in line}

#file wrong word
with open('wrong-word.txt', 'r', encoding='utf8') as file:
    wrong_lst = file.read().split('\n')

#file stopword
with open('vietnamese-stopwords.txt', 'r', encoding='utf8') as file:
    stopwords_lst = file.read().split('\n')



phrases_to_join = [ "dưỡng ẩm", "nhạy cảm", "làm sạch", "cấp ẩm", "tái tạo da",
    'thấm nhanh', 'không ảnh hưởng', 'sáng da', 'sạch sâu',
    'mờ thâm', 'da dầu', 'chống nắng', 'trị mụn',
    'dịu nhẹ', 'lỗ chân lông', 'hàng ngày', 'mịn da',
     'ngăn ngừa', 'lão hóa', 'dạng nước', 'kiềm dầu', 'ngừa mụn', 'bọt mịn',
    'giảm mụn', 'da mụn', 'giảm nhờn','khô da',
    'dịu nhẹ', 'se lỗ chân lông', 'tẩy tế bào chết',
   'tẩy da chết', 'chống bụi', 'lột mụn', 'dán mũi', 'tạo bọt',
    'nâng tông',  'quầng thâm', 'bọng mắt', 'thẩm thấu', 'nhanh thấm', 'cay mắt', 'chính hãng',  'hiệu quả', 'hối hận', 'thất vọng',
    'tệ hại', 'kinh khủng',  'không ưng ý', 'không được tốt', 'kinh khủng', 'mùi nồng', 'không dùng được',
    'không nên', 'không đỡ','bị mụn', 'cay mắt', 'hối hận',  'không xứng đáng', 'châm chích', 'đừng nên',
    'bết rít', 'âm điểm', 'sai lầm', 'khó chịu', 'khô da', 'phí tiền', 'kém bền', 'da sần', 'mùi cồn',
    'bực mình', 'không đạt', 'nhờn rít','hết hạn', 'ph cao', 'vô dụng', 'không kiềm dầu', 'trắng bệch',
    'khó thấm', 'dị ứng', 'giận dữ', 'mệt mỏi',  'cải thiện', 'sữa rửa mặt', 'tẩy sạch', 'dịu nhẹ',
    'chống nắng', 'nâng tông', 'lên tông', 'hiệu quả', 'thư giãn']

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from underthesea import word_tokenize

# Hàm xử lý văn bản
def process_text_pipeline(text, emoji_dict, teen_dict, wrong_lst, stopwords, phrases_to_join):
    # Bước 1: Chuyển văn bản về chữ thường
    text = text.lower()

    # Bước 2: Thay thế emoji
    text = ''.join(emoji_dict.get(char, char) for char in text)

    # Bước 3: Thay thế teen slang
    text = ' '.join(teen_dict.get(word, word) for word in text.split())

    # Bước 4: Loại bỏ các từ không mong muốn
    text = ' '.join(word for word in text.split() if word not in wrong_lst)

    # Bước 5: Ghép các cụm từ
    for phrase in phrases_to_join:
        text = text.replace(phrase, '_'.join(phrase.split()))

    # Bước 6: Loại bỏ stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords)

    # Bước 7: Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text




# Hàm đếm số lượng từ
def count_words(text, word_list):
    # Tạo regex để đếm số lượng từ trong word_list
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in word_list) + r')\b'
    return len(re.findall(pattern, text))

# Hàm vector hóa văn bản
def vectorize_text(data, column):
    vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.9)
    train_vec  = vectorizer.fit_transform(pre_df['word_pro3'])
    X = vectorizer.transform(data[column])
    return X, vectorizer



# Hàm xử lý và dự đoán cảm xúc
def analyze_file(lines, emoji_dict, teen_dict, wrong_lst, stopwords, phrases_to_join, word_nev, word_pov, model):
    if isinstance(lines, str):
        lines = [lines]
    # Tạo DataFrame từ các dòng
    data = pd.DataFrame({'text': [line.strip() for line in lines]})
    
    # Đảm bảo cột 'text' tồn tại
    if 'text' not in data.columns:
        raise ValueError("The input data must contain a 'text' column.")

    # Bước 2: Xử lý văn bản
    data['processed_text'] = data['text'].apply(
        lambda x: process_text_pipeline(x, emoji_dict, teen_dict, wrong_lst, stopwords, phrases_to_join)
    )

    # Bước 3: Đếm từ tiêu cực và tích cực
    data['negative_count'] = data['processed_text'].apply(lambda x: count_words(x, word_nev))
    data['positive_count'] = data['processed_text'].apply(lambda x: count_words(x, word_pov))

    # Bước 4: Tính độ dài văn bản
    data['text_length'] = data['processed_text'].apply(len)

    # Bước 5: Vector hóa văn bản
    X_text, vectorizer = vectorize_text(data, 'processed_text')

    # Bước 6: Kết hợp các đặc trưng
    import numpy as np
    additional_features = data[['negative_count', 'positive_count']].values
    X = np.hstack((X_text.toarray(), additional_features))

    # Bước 7: Dự đoán cảm xúc
    data['prediction'] = model.predict(X)

    # Trả về DataFrame kết quả
    return data[['text', 'prediction']]




#1. load model
from joblib import load
loaded_model_random = load('model_random_weight.joblib')

#4. Evaluate model
y_pred = loaded_model_random.predict(X_test)
score_train = loaded_model_random.score(X_train,y_train)
score_test = loaded_model_random.score(X_test,y_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])

cr = classification_report(y_test, y_pred)

y_prob = loaded_model_random.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:,1])


#Product Visualization
def analyze_product_by_code(ma_san_pham):
    # Kiểm tra mã sản phẩm có tồn tại trong hệ thống không
    product_data = san_pham[san_pham['ma_san_pham'] == ma_san_pham]['ten_san_pham'].values

    if not product_data:
        st.write(f"Mã sản phẩm '{ma_san_pham}' không tồn tại trong hệ thống.")
        return

    # Lấy tên sản phẩm từ mã sản phẩm
    product_name = product_data[0]
    st.write(f"Tên sản phẩm: {product_name}")

    # Lọc dữ liệu nhận xét tích cực và tiêu cực
    filtered_df_pos = pre_df[(pre_df['star_cleaned'] == 5) & (pre_df['ma_san_pham'] == ma_san_pham)]
    filtered_df_neg = pre_df[(pre_df['star_cleaned'] == 1) & (pre_df['ma_san_pham'] == ma_san_pham)]

    # Số lượng nhận xét tích cực và tiêu cực
    positive_count = len(filtered_df_pos)
    negative_count = len(filtered_df_neg)

    st.write(f'Số lượng nhận xét tích cực: {positive_count}')
    st.write(f'Số lượng nhận xét tiêu cực: {negative_count}')

    # 1. Tính toán số sao bình quân
    avg_rating = pre_df[pre_df['ma_san_pham'] == ma_san_pham]['star_cleaned'].mean()
    st.write(f"Số sao bình quân: {avg_rating:.2f}")

    # 2. Thời gian của các bình luận
    # Đảm bảo chuyển đổi ngày tháng đúng định dạng
    filtered_df_pos['ngay_binh_luan'] = pd.to_datetime(filtered_df_pos['ngay_binh_luan'], dayfirst=True, errors='coerce')
    filtered_df_neg['ngay_binh_luan'] = pd.to_datetime(filtered_df_neg['ngay_binh_luan'], dayfirst=True, errors='coerce')

    # Loại bỏ giá trị NaT
    filtered_df_pos = filtered_df_pos.dropna(subset=['ngay_binh_luan'])
    filtered_df_neg = filtered_df_neg.dropna(subset=['ngay_binh_luan'])

    # Tạo danh sách ngày bình luận
    comment_dates = (
        pd.to_datetime(filtered_df_pos['ngay_binh_luan']).tolist() +
        pd.to_datetime(filtered_df_neg['ngay_binh_luan']).tolist()
    )

    # Kiểm tra danh sách không rỗng trước khi lấy min/max
    if comment_dates:
        st.write(f"Thời gian bình luận: {min(comment_dates).strftime('%Y-%m-%d')} đến {max(comment_dates).strftime('%Y-%m-%d')}")
    else:
        st.write("Không có dữ liệu ngày bình luận hợp lệ.")


    # 3. Tính toán số lượng bình luận theo giờ trong ngày
    filtered_df = pre_df[pre_df['ma_san_pham'] == ma_san_pham]
    filtered_df['gio_binh_luan'] = pd.to_datetime(filtered_df['gio_binh_luan'])
    hour_counts = filtered_df.groupby(filtered_df['gio_binh_luan'].dt.hour).size().reset_index(name='count')

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo fig và ax
    ax.bar(hour_counts['gio_binh_luan'], hour_counts['count'], color='skyblue')
    ax.set_xlabel('Giờ trong ngày', fontsize=14)
    ax.set_ylabel('Số lượng bình luận', fontsize=14)
    ax.set_title(f'Số lượng bình luận theo giờ trong ngày', fontsize=18,fontweight='bold')
    ax.set_xticks(hour_counts['gio_binh_luan'])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)  # Hiển thị fig

    # 4. Tạo WordCloud cho các nhận xét tích cực và tiêu cực
    def create_wordcloud(df, sentiment):
        comments = " ".join(df['joined_positive_words'].dropna()) if sentiment == "positive" else " ".join(df['joined_negative_words'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments)
        fig, ax = plt.subplots(figsize=(10, 5))  # Tạo fig và ax
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"WordCloud cho nhận xét {sentiment}", fontsize=18, fontweight='bold')  
        st.pyplot(fig)  # Hiển thị fig

    # Hiển thị wordcloud cho nhận xét tích cực và tiêu cực
    create_wordcloud(filtered_df_pos, "positive")
    create_wordcloud(filtered_df_neg, "negative")




#GUI
st.title("Data Science Project")
st.write("## Sentiment Analysis")

menu = ["Project Info",  "Build Model", "New Prediction"]
# Sidebar Menu
choice = st.sidebar.selectbox('Menu', menu)
# Sidebar Content
st.sidebar.markdown(
    """
    ### **Thông tin dự án**
    - **Thành viên thực hiện**
      - 🟢 **Phan Thị Thu Hạnh**
      - 🟡 **Nguyễn Hải Yến**
    
    - **Giảng viên hướng dẫn**
      - 🟣 **Cô Khuất Thùy Phương**
    
    - **Thời gian báo cáo**
      - 📅 **14/12/2024**
    """,
    unsafe_allow_html=True
)


if choice == 'Project Info':
    tab1, tab2, tab3, tab4 = st.tabs(['Business Overall', 'Product Info', 'Algorithms Info', 'About Us'])
    with tab1:
        st.image('hasaki.jpg')
        st.subheader('Hasaki - chuỗi hệ thống phân phối mỹ phẩm với hơn 500 thương hiệu')

        st.write(""" Hasaki là một chuỗi hệ thống phân phối mỹ phẩm đang được phát triển tại Việt Nam. Doanh nghiệp đang ngày càng mở rộng trải dài khắp toàn quốc. Bên cạnh đó, Hasaki đang chú trọng nâng cấp phát triển hệ thống bán hàng trực tuyến hasaki.vn để tăng trải nghiệm người dùng. Mỗi ngày Hasaki nhận về hàng ngàn lượt đánh giá và tương tác của khách hàng đối với sản phẩm của mình.
                """)
        
        #tạo 2 hình
        col1, col2 = st.columns(2)
        with col1:
            st.image('tab1_2.png', caption='Ứng dụng mua sắm trực tuyến hasaki.vn', use_container_width=True)
        with col2:
            st.image('tab1_1.png', caption='Hệ thống mỹ phẩm 76 chi nhánh trải dài 27 tỉnh thành', use_container_width=True)
        st.write("""Việc quản lý và thu thập ý kiến đánh giá từ khách hàng là yếu tố then chốt trong việc cải thiện sản phẩm, mở rộng tệp khách hàng và xây dựng thương hiệu.
Nhờ vào hệ thống trực tuyến, Hasaki có thể thu thập ý kiến một cách khách quan từ người dùng thực tế. 
                 Những dữ liệu giá trị này giúp doanh nghiệp phân tích sâu hơn, từ đó phát triển hệ thống dự đoán tâm lý khách hàng trong tương lai một cách chính xác và hiệu quả hơn.""")
        st.image('tab1_3.png', caption='Mỗi ngày hasaki.vn thu về hàng ngàn lượt tương tác từ khách hàng')

        st.write("##### Phương án đề xuất")
        st.write('Phát triển hệ thống dự đoán tâm tư khách hàng thông qua những dữ liệu đánh giá hiện có bằng Machine Learning.')

    with tab2:
        st.image('tab2_1.png')
        st.subheader("Khám phá sản phẩm thông qua tương tác từ khách hàng")
        st.write(""" Phân tích cảm xúc của khách hàng thông qua những bình luận về sản phẩm 
                là một trong những cách giúp cho HASAKI hiểu rõ hơn về khách hàng của mình 
                cũng như những đánh giá khách quan nhất và chính xác nhất cho sản phẩm của mình 
                thông qua người dùng thực tế. Từ đó doanh nghiệp có thể có những kế hoạch
                phát triển sản phẩm cũng như mở rộng tệp khách hàng hiện có của mình""")
        col1, col2= st.columns(2)
        with col1:
            # Vẽ biểu đồ
            top_10_products = merged_data['ten_san_pham'].value_counts().nlargest(10)
            # Cắt ngắn tên sản phẩm nếu cần (giới hạn 20 ký tự)
            top_10_products.index = top_10_products.index.str.slice(0, 20)
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(12, 5))
            top_10_products.plot(kind='bar', color='purple', ax=ax)
            # Thiết lập nhãn và tiêu đề
            ax.set_xlabel('Tên sản phẩm')
            ax.set_ylabel('Số lượng bình luận')
            ax.set_title('Top 10 sản phẩm có nhiều bình luận nhất')
            # Xoay nhãn trục X
            plt.xticks(rotation=45, ha='right')
            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

        with col2:

            # Vẽ biểu đồ phân bố của số sao (star_cleaned)
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']  # Gradient xanh - cam - tím
            # Số sao và số lượng
            star_counts = data['so_sao'].value_counts().sort_index()
            star_counts.plot(kind='bar', color=colors[:len(star_counts)], ax=ax)
            # Thiết lập nhãn và tiêu đề
            ax.set_xlabel('Số sao đánh giá')
            ax.set_ylabel('Số lượng')
            ax.set_title('Phân bố của số sao đánh giá')
            # Cố định nhãn trục X
            ax.set_xticks(range(len(star_counts)))
            ax.set_xticklabels(star_counts.index, rotation=0)
            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            # Vẽ biểu đồ thời gian bình luận
            data['ngay_binh_luan'] = pd.to_datetime(data['ngay_binh_luan'], errors='coerce')
            # Loại bỏ giá trị thiếu
            danh_gia = data.dropna(subset=['ngay_binh_luan'])
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(15, 5))
            danh_gia['ngay_binh_luan'].dt.date.value_counts().sort_index().plot(kind='line', color='orange', ax=ax)
            # Thiết lập nhãn và tiêu đề
            ax.set_xlabel('Ngày bình luận')
            ax.set_ylabel('Số lượng bình luận')
            ax.set_title('Tần suất bình luận theo ngày')

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

        with col4:
            #Vẽ biểu đồ bình luận theo giờ
            danh_gia['gio_binh_luan'] = pd.to_datetime(danh_gia['gio_binh_luan'], format='%H:%M', errors='coerce')
            # Loại bỏ giá trị thiếu
            danh_gia = danh_gia.dropna(subset=['gio_binh_luan'])
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(10, 5))
            danh_gia['gio_binh_luan'].dt.hour.value_counts().sort_index().plot(kind='bar', color='green', ax=ax)
            # Thiết lập nhãn và tiêu đề
            ax.set_xlabel('Giờ bình luận')
            ax.set_ylabel('Số lượng bình luận')
            ax.set_title('Tần suất bình luận theo giờ trong ngày')

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)


        st.subheader('Phân tích nội dung bình luận')
        st.write('Hiểu rõ hơn nội dung các bình luận thông qua việc hiển thị các từ khóa có ý nghĩa xuất hiện phổ biến nhất trong tất cả các bình luận')

        # Trực quan nội dung từ khóa sản phẩm
        text = " ".join(pre_df['word_pro3'])
        # Tạo WordCloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOPWORDS,
            colormap='viridis'
        ).generate(text)
        # Hiển thị WordCloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  # Tắt trục tọa độ

        # Hiển thị WordCloud trên Streamlit
        st.pyplot(fig)

        #Trực quan từ khóa tích cực, tiêu cực
        col5, col6 = st.columns(2)

        with col5:
            
            text = " ".join(data_visual['joined_positive_words'])
            # Tạo WordCloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=STOPWORDS,
                colormap='viridis'
            ).generate(text)
            st.write("**Biểu đồ WordCloud của các từ tích cực**")
            # Hiển thị WordCloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')  # Tắt trục tọa độ

            # Hiển thị WordCloud trên Streamlit
            st.pyplot(fig)


        with col6:
            text = " ".join(data_visual['joined_negative_words'].astype(str))
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=STOPWORDS,
                colormap='viridis'
            ).generate(text)

            # Tiêu đề cho WordCloud
            st.write("**Biểu đồ WordCloud của các Từ Tiêu Cực**")

            # Hiển thị WordCloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')  # Tắt trục tọa độ

            # Hiển thị WordCloud trên Streamlit
            st.pyplot(fig)


        st.subheader('Khám phá thông tin chi tiết từng sản phẩm')
        # Tạo text_input để nhập liệu
        ma_san_pham_input = st.text_input("Nhập mã sản phẩm:")

        # Tạo selectbox với danh sách mã sản phẩm
        ma_san_pham_options = san_pham['ma_san_pham'].unique().tolist()
        ma_san_pham_select = st.selectbox("Hoặc chọn từ danh sách:", ma_san_pham_options)

        # Kiểm tra xem người dùng đã nhập liệu hay chọn từ danh sách
        if ma_san_pham_input:
            ma_san_pham = int(ma_san_pham_input)
            st.write("Bạn đã nhập:", ma_san_pham)
        else:
            ma_san_pham = ma_san_pham_select
            st.write("Bạn đã chọn:", ma_san_pham)

        # Nút "Phân tích"
        if st.button("Phân tích"):
            if ma_san_pham_input or ma_san_pham_select:  # Kiểm tra cả hai trường hợp
                # Gọi hàm phân tích
                analyze_product_by_code(ma_san_pham)
            else:
                st.warning("Vui lòng nhập mã sản phẩm.")
        
    with tab3:
        st.image('tab3_2.png')
        st.subheader('Random Forest - Thuật toán xây dựng mô hình trong Machine Learning')
        st.write('Random Forest là một thuật toán học máy mạnh mẽ và phổ biến thuộc nhóm ensemble learning (học kết hợp). Đây là một thuật toán phân loại và hồi quy được phát triển bởi Leo Breiman vào năm 2001. Random Forest sử dụng nhiều cây quyết định (decision trees) để tạo ra một mô hình mạnh mẽ hơn và giảm thiểu hiện tượng overfitting (mô hình quá khớp với dữ liệu huấn luyện)')
        st.image('tab3_4.png')
        st.write('## **So sánh độ chính xác và thời gian dự đoán so với các mô hình khác**')
        col1, col2 = st.columns(2)
        with col1:
            st.image('tab3_model_compare1.png')
            st.image('tab3_model_compare2.png')
        with col2:
            st.image('tab3_model_compare3.png',width=300)
            st.image('tab3_model_compare4.png', width=300)

        st.subheader('Random Forest là thuật toán phù hợp để xây dựng mô hình')
        st.write("""
                 1. Bộ dữ liệu train sử dụng không quá lớn\n
                 2. Mô hình có độ chính xác cao và thời gian xử lý nhanh\n
                 3. Mô hình giảm overfitting\n
                 4. Khả năng chống nhiễu tốt 
""")
    with tab4:
        st.subheader('Đồ Án Tốt Nghiệp Data Science & Machine Learning')
        st.write('**Trung Tâm Tin Học - Trường Đại Học Khoa Học Tự Nhiên**')
        st.write('Thời gian báo cáo: 14/12/2024')
        st.write("Giảng viên hướng dẫn: Cô Khuất Thùy Phương")
        st.write("""Học viên thực hiện: Phan Thị Thu Hạnh & Nguyễn Hải Yến""")
        st.write('Nhóm: Nhóm 4')
        st.write('Khóa: DL07_K299')
        st.markdown("---")
        st.write('## Thông tin học viên')
        col1, col2 = st.columns(2)
        with col1:
            st.image('tab4_4.png', caption='Học viên: Phan Thị Thu Hạnh')
        with col2:
            st.image('tab4_2.png', caption='Học viên: Nguyễn Hải Yến')
        st.markdown("---")
        st.markdown("*Học viên chúng em xin gửi lời cảm ơn trân trọng nhất đến Cô Khuất Thùy Phương đã tận tình hướng dẫn để chúng em có thể hoàn thành Đồ án tốt nghiệp này!*")
            



elif choice== 'Build Model':
    st.subheader('Xây dựng mô hình dự đoán cảm xúc khách hàng với thuật toán Random Forest')

    st.write('#### 1. Tổng quan dữ liệu')
    st.write('Dữ liệu được ghi lại trực tiếp từ trang bán hàng trực tuyến Hasaki.vn, nội dung bình luận và số sao được đánh giá là hai dữ liệu quan trọng được sử dụng chính trong việc xây dựng mô hình dự đoán.')
    st.dataframe(data[['noi_dung_binh_luan', 'so_sao']].head(3))
    st.dataframe(data[['noi_dung_binh_luan', 'so_sao']].tail(3))
    # Vẽ biểu đồ phân bố của số sao (star_cleaned
    st.dataframe(data['so_sao'].value_counts())

    st.write('Các thư viện sử dụng để làm sạch dữ liệu')
    st.code('''
            import pandas as pd
            import regex
            import string
            from underthesea import word_tokenize, pos_tag, sent_tokenize
            from wordcloud import WordCloud, STOPWORDS
            import time
            import re''')
    

    st.write('Dữ liệu bình luận đã được làm sạch, dữ liệu đánh giá số sao được chuyển thành 2 loại 1 sao và 5 sao')
    st.dataframe(pre_df[['word_pro3', 'star_cleaned']].head(3))
    st.dataframe(pre_df[['word_pro3', 'star_cleaned']].tail(3))
    st.dataframe(pre_df['star_cleaned'].value_counts())

    st.write('Sử dụng TF-IDF để mã hóa bộ dữ liệu')
    st.code('''def vectorize_text(data, column):
                vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.9)
                train_vec  = vectorizer.fit_transform(pre_df['word_pro3'])
                X = vectorizer.transform(data[column])
                return X, vectorizer''')
    
    
    st.write('##### 2. Visualize Positive & Negative')
    #Trực quan từ khóa tích cực, tiêu cực
    col5, col6 = st.columns(2)

    with col5:
        
        text = " ".join(data_visual['joined_positive_words'])
        # Tạo WordCloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOPWORDS,
            colormap='viridis'
        ).generate(text)
        st.write("**Biểu đồ WordCloud của các từ tích cực**")
        # Hiển thị WordCloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  # Tắt trục tọa độ

        # Hiển thị WordCloud trên Streamlit
        st.pyplot(fig)


    with col6:
        text = " ".join(data_visual['joined_negative_words'].astype(str))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOPWORDS,
            colormap='viridis'
        ).generate(text)
        st.write("**Biểu đồ WordCloud của các từ tiêu cực**")
        # Hiển thị WordCloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  # Tắt trục tọa độ
        st.pyplot(fig)


    st.write('##### 3. Xây dựng mô hình với thuật toán Random Forest')
    st.code('from sklearn.ensemble import RandomForestClassifier')
    st.write('Do dữ liệu bị mất cân bằng, nên khi xây dựng mô hình cần điều chỉnh lại trọng số')
    st.code('''
            class_weights = {1: 8, 5: 1}  # Tăng trọng số lớp 1
            model_random = RandomForestClassifier(class_weight = class_weights, random_state = 42)
            ''')

    st.write('##### 4. Đánh giá mô hình')
    st.code("Score train:" + str(round(score_train,2)) + " và Score Test:" + str(round(score_test,2)))
    st.code('Accuracy:' + str(round(acc,2)))

    st.write('##### confusion matrix: ')
    st.image("confusion_matrix.png", width=500)
    st.write('''
             Tổng số dự đoán chính xác: 236 + 3491 = 3727.\n
             Tổng số dự đoán sai: 31 + 159 = 190.\n
             Tỷ lệ dự đoán đúng trên tổng dự đoán = 95.17%\n
             Với số liệu trên cho thấy mô hình dự đoán tốt đặc biệt ở lớp 5, ma trận nhầm lớp 1 sang lớp 5 không quá cao, có thể chấp nhận được.''')
    st.write('##### Classification report:')
    st.code(cr)

    #calculate roc curve
    st.write('##### ROC curve')
    st.image("roc.png", width=500)
    st.code('ROC AUC score:' + str(round(roc,2)))
    st.write('Với giá trị 0.97 rất gần với giá trị tối đa là 1.0, nghĩa là mô hình có khả năng phân biệt giữa các lớp tích cực (positive) và tiêu cực (negative) một cách chính xác trong phần lớn các trường hợp')


    st.write('##### 5. Kết luận:')
    st.write('Mô hình dự đoán sử dụng thuật toán Random Forest của Machine Learning đã được xây dựng và sẵn sàng đưa vào sử dụng')

elif choice == 'New Prediction':

    st.subheader('Dự đoán cảm xúc khách hàng với mô hình Machine Learning - Thuật toán Random Forest')
    flag = False
    lines = None
    # Chọn cách nhập dữ liệu
    type = st.radio('Upload data or Input data?', options=('Upload', 'Input'))

    if type == 'Upload':
        # Upload file
        upload_file_1 = st.file_uploader('Choose a file:', type=['txt', 'csv'])
        if upload_file_1 is not None:
            lines = pd.read_csv(upload_file_1, header=None)
            st.dataframe(lines)  # Hiển thị nội dung file đã upload
            lines = lines[0]  # Lấy cột đầu tiên làm nội dung dự đoán
            flag = True

    elif type == 'Input':
    # Nhập trực tiếp nội dung
        lines = st.text_area(label='Input your content:')
        if lines.strip() != '':  # Kiểm tra nếu người dùng nhập nội dung
            flag = True

    # Thực hiện dự đoán nếu có dữ liệu
    if flag:
        st.write('Content:')
        st.code(lines)  # Hiển thị nội dung đã nhập hoặc tải lên
        result = analyze_file(
            lines=lines,
            emoji_dict=emoji_dict,
            teen_dict=teen_dict,
            wrong_lst=wrong_lst,
            stopwords=stopwords_lst,
            phrases_to_join=phrases_to_join,
            word_nev=word_nev,
            word_pov=word_pov,
            model=loaded_model_random
        )

        st.code('New prediction (1: Tiêu cực, 5: Tích cực)')
        st.dataframe(result)  # Hiển thị kết quả dự đoán
