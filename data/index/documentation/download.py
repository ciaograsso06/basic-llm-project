import requests

ZABBIX_PDF_URL = "https://www.zabbix.com/documentation/6.4/downloads/Zabbix_Documentation_6.4.en.pdf"

def download_pdf(url, path):
    print("Baixando PDF...")
    response = requests.get(url)
    with open(path, "wb") as file:
        file.write(response.content)
    print(f"PDF baixado em: {path}")
    

if __name__ == "__main__":
    download_pdf(ZABBIX_PDF_URL, "zabbix_documentation.pdf")