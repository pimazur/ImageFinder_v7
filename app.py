import streamlit as st
from dotenv import dotenv_values
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
import base64



env = dotenv_values(".env")

if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']


EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 3072

QDRANT_COLLECTION_NAME = "image_descriptions"

IMAGES_PATH = Path("images")

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

#
# DB
#
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env['QDRANT_URL'], 
    api_key=env['QDRANT_API_KEY'],
)

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_description_to_db(description, file_name):
    qdrant_client = get_qdrant_client()
    first_available_id = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True
        ).count + 1
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=first_available_id,
                vector=get_embedding(description),
                payload={'file_name': file_name}
            )
        ]
    )

def search_descriptions_in_db(user_search):
    qdrant_client = get_qdrant_client()
    
    return qdrant_client.search(
    collection_name=QDRANT_COLLECTION_NAME,
    query_vector=get_embedding(user_search),
    limit=1,
    )[0].payload['file_name']

#
# Image to text
#
def prepare_image_for_open_ai(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:image/png;base64,{image_data}"

def describe_image(image_path):
    openai_client = get_openai_client()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Stwórz opis obrazka w języku polskim, jakie widzisz tam elementy?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_image_for_open_ai(image_path),
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

def save_image(file_path, uploaded_file):
    with open(file_path, 'wb') as file:
        file.write(uploaded_file.getvalue())

#
# MAIN
#
st.set_page_config(page_title='Image Finder')

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.header('Image Finder')
        st.info("Wprowadź swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()


IMAGES_PATH.mkdir(exist_ok=True)
assure_db_collection_exists()


st.header('Image Finder')
st.write(
    'Witaj w Image Finder! Możesz tu zapisywać ' \
    'dowolne obrazy, a następnie wyszukiwać ' \
    'je po powiązanych z nimi hasłach ' \
    'z pomocą sztucznej inteligencji.'
)

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False


add_tab, search_tab = st.tabs(['Załaduj zdjęcie', 'Wyszukaj zdjęcie'])
with add_tab:
    uploaded_file = st.file_uploader('Przeciągnij i upuść zdjęcie:', type=['jpg', 'jpeg', 'png'])
    
    if not st.session_state['uploaded_image'] == uploaded_file:
        st.session_state['uploaded_image'] = uploaded_file
    
    if st.session_state['uploaded_image']:

        button = st.button('Zapisz obraz', disabled=st.session_state['button_clicked'])
        if button:
            st.session_state['button_clicked'] = True
            st.rerun()

        if st.session_state['button_clicked']:
            file_path = IMAGES_PATH / uploaded_file.name
            if file_path.exists():
                st.error('Podany plik już istnieje. Zmień nazwę, aby zapisać.')
                st.session_state['button_clicked'] = False
            else:
                spinner_emoji = st.markdown(
                    'Poczekaj chwilę...  ' \
                    '<span style="font-size: 24px;">☕</span>' \
                    '<span style="font-size: 24px;">💆‍♂️</span>'
                    , unsafe_allow_html=True
                )
                save_image(file_path, uploaded_file)
                image_description = describe_image(file_path)
                add_description_to_db(image_description, uploaded_file.name)
                spinner_emoji.empty()
                success_message = st.success('Plik zapisano.')
                st.session_state['button_clicked'] = False
                st.rerun()

with search_tab:
    search_image = st.text_input('Wyszukaj obraz')
    if search_image:
        result = search_descriptions_in_db(search_image)
        st.image(
            image=IMAGES_PATH / result,
            use_container_width=True,
        )