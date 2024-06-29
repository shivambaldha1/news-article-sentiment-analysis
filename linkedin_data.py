import requests

url = "https://linkedin-data-api.p.rapidapi.com/get-company-details"

def scrape_company_data(company_name):
    querystring = {"username":company_name}

    headers = {
        "x-rapidapi-key": "fef0f36761msh7f7079330e839fbp14e3ebjsna3e9db8b6c39",
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    return (response.json())