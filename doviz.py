#Oncelikle Google Cloud proje, bucket bilgilerini almak ve ayarları yapmanız gerekir.

#Python Reasoning Engine paketini kullanmak için Vertex AI SDK indirilmesi
pip install google-cloud-aiplatform[reasoningengine,langchain]

#Reasoning Engine SDK indirmek için aşağıdaki kod kullanılır
import vertexai
from vertexai.preview import reasoning_engines

vertexai.init(
    project="multimodal1-430318",
    location="us-central1",
    staging_bucket="gs://aisprint_langchain",
)

#Gemini modeli seçimi: guncel pro veya flash modeli seçilebilir.
model = "gemini-1.5-flash-001"

#Guvenlik Ayarları, istege bagli, yoksa standart ayarlar kullanılır.
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

#Model parametreleri belirlenmesi, bu da istege bagli, yapılmazsa standart ayarlar kullanılır.
model_kwargs = {
    # temperature (float): Token seçiminde rastgelelik derecesini kontrol eden örnekleme sıcaklığı.
    "temperature": 0.28,
    # max_output_tokens (int): Token sınırı, bir promptdan gelen maksimum metin çıktısı miktarını belirler.
    "max_output_tokens": 1000,
    # top_p (float): Tokenler, olasılıklarının toplamı top-p değerine eşit olana kadar en olası olandan en az olası olana doğru seçilir.
    "top_p": 0.95,
    # top_k (int): Bir sonraki token, en olası top-k işaretçi arasından seçilir. Bu, tüm model sürümleri tarafından desteklenmez.
    "top_k": None,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): İçerik oluşturmak için kullanılacak güvenlik ayarları.
    # (önceki adımı kullanarak önce güvenlik ayarlarınızı oluşturmalısınız).
    
    "safety_settings": safety_settings,
}

#Artık model yapılandırmalarını kullanarak bir LangchainAgent oluşturabilir ve sorgulayabilirsiniz:

agent = reasoning_engines.LangchainAgent(
    model=model,                # Gerekli
    model_kwargs=model_kwargs,  # İstege bagli
)

#Yanıt, bir Python sözlüğü olacaktır ve guncel bilgiyi veremeyecegini belirtecektir.
response = agent.query(input="1 USD kaç TRY'dir?")
response
#{'input': "1 USD kaç TRY'dir?",'output': 'Maalesef, ben gerçek zamanlı bilgi sağlayamıyorum. 
#Dolayısıyla, güncel USD/TRY kurunu bilmiyorum.\n\n
#Güncel döviz kuru için bir finans web sitesi veya döviz çevirici kullanmanızı öneririm. \n'}

#model= ve model_kwargs= için desteklenen değerler kümesi her sohbet modeli için farklıdır, 
#bu nedenle ayrıntılar için ilgili belgelerine başvurmanız gerekir.
agent = reasoning_engines.LangchainAgent(
    model=model,                # Gerekli
    model_kwargs=model_kwargs,  # İstege baglı.
)

#Modelinizi tanımladıktan sonraki adım, modelinizin Reasoning(muhakeme) için kullandığı araçları(Tools) tanımlamaktır. 
#Bir araç (Tool), bir LangChain aracı veya bir Python fonksiyonu olabilir.

def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):
    """İki para birimi arasındaki döviz kurunu belirtilen tarihte çevirir .

Döviz kuru verilerini elde etmek için Frankfurter API'sini (https://api.frankfurter.app/) kullanır.

Argümanlar:
currency_from: Temel para birimi (3 harfli para birimi kodu).
Varsayılan olarak "USD" (ABD Doları).
currency_to: Hedef para birimi (3 harfli para birimi kodu).
Varsayılan olarak "EUR" (Euro).
currency_date: Döviz kurunun alınacağı tarih.
En son döviz kuru verileri için varsayılan olarak "latest".
Geçmiş oranlar için YYYY-AA-GG biçiminde belirtilebilir.

Sonuç şu şekilde olur:
dict: Döviz kuru bilgilerini içeren bir sözlük.
Örnek: {"amount": 1.0, "base": "USD", "date": "2023-11-24",
"rates": {"EUR": 0.95534}}"""
    """
    import requests
    response = requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    )
    return response.json()
    
#"Uygulamanızda kullanmadan önce fonksiyonu test etmek için aşağıdaki kodu çalıştırın:"
get_exchange_rate(currency_from="USD", currency_to="TRY")

#Aracı(Tool) LangchainAgent şablonunun içinde kullanmak için, tools= argümanı altındaki araçlar listesine ekleyeceksiniz:
agent = reasoning_engines.LangchainAgent(
    model=model,                # Gerekli
    tools=[get_exchange_rate],  # Istege bagli
    model_kwargs=model_kwargs,  # Istege bagli
)

#Yanıt, aşağıdakine benzer bir sözlük olacaktır:
response = agent.query(
    input="1 USD kaç TRY'dir?"
)
response
#{'input': "Y1 USD kaç TRY'dir?", 'output': "1 USD, 34.181 TRY'dir. \n"}
