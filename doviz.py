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
#Yanıt, bir Python sözlüğü olacaktır ve guncel bilgiyi veremeyecegini belirtecektir.
agent = reasoning_engines.LangchainAgent(
    model=model,                # Required.
    model_kwargs=model_kwargs,  # Optional.
)

response = agent.query(input="What is the exchange rate from US dollars to Swedish currency?")
response

agent = reasoning_engines.LangchainAgent(
    model=model,                # Gerekli
    model_kwargs=model_kwargs,  # İstege baglı.
)

def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):
    """Retrieves the exchange rate between two currencies on a specified date.

    Uses the Frankfurter API (https://api.frankfurter.app/) to obtain
    exchange rate data.

    Args:
        currency_from: The base currency (3-letter currency code).
            Defaults to "USD" (US Dollar).
        currency_to: The target currency (3-letter currency code).
            Defaults to "EUR" (Euro).
        currency_date: The date for which to retrieve the exchange rate.
            Defaults to "latest" for the most recent exchange rate data.
            Can be specified in YYYY-MM-DD format for historical rates.

    Returns:
        dict: A dictionary containing the exchange rate information.
            Example: {"amount": 1.0, "base": "USD", "date": "2023-11-24",
                "rates": {"EUR": 0.95534}}
    """
    import requests
    response = requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    )
    return response.json()

get_exchange_rate(currency_from="USD", currency_to="SEK")

agent = reasoning_engines.LangchainAgent(
    model=model,                # Required.
    tools=[get_exchange_rate],  # Optional.
    model_kwargs=model_kwargs,  # Optional.
)

response = agent.query(
    input="What is the exchange rate from US dollars to Swedish currency?"
)
response
