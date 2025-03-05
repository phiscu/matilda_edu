import requests
import hashlib
import json

class ResourceSpace:
    
    #api_base_url: the base url of your repository (e.g. rs.cms.hu-berlin.de/demo/api/?)
    #private_key: your account's private key (you may get it from your administrator)
    #user: your user name
    def __init__(self, api_base_url, user, private_key):
        self.api_base_url = api_base_url
        self.user = user
        self.private_key = private_key
        
    #uses get_resource_data of the resourcespace API: https://www.resourcespace.com/knowledge-base/api/get_resource_data
    #returns a response object https://www.w3schools.com/python/ref_requests_response.asp
    def do_request(self, query):
        query = 'user=' + self.user + '&' + query
        sign = hashlib.sha256((self.private_key + query).encode()).hexdigest()
        url = self.api_base_url + query + '&sign=' + sign
        return requests.get(url)
    
    #uses do_search of the resourcespace API: https://www.resourcespace.com/knowledge-base/api/do_search
    #search_term: your search term (you can also use resourcespace's special search terms: https://www.resourcespace.com/knowledge-base/user/special-search-terms)
    #returns a python array of resource descriptions as python objects
    def do_search(self, search_term):

        #generating query
        query = 'function=do_search&search=' + requests.utils.quote(str(search_term))
    
        response = self.do_request(query)
    
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            return None


    #uses get_resource_data of the resourcespace API: https://www.resourcespace.com/knowledge-base/api/get_resource_data
    #ref: the id of the resource you want to get the data of
    #returns a python object
    def get_resource_data(self, ref):
    
        #generating query
        query = 'function=get_resource_data&resource=' + str(ref)
        #request
        response = self.do_request(query)
        
        #returning result
        if(response.status_code == 200):
            return json.loads(response.text)
        else:
            return None

    #uses get_resource_path of the resourcespace API: https://www.resourcespace.com/knowledge-base/api/get_resource_path
    #ref: the id of the resource you want to get the content of or an array of resource ids
    #returns the content in bytes if the request was successful else None
    def get_resource_file(self, ref, ext=""):
        
        if(ext==""):
            resource_data = self.get_resource_data(ref)
            ext = resource_data["file_extension"];

        #generating query
        query = 'function=get_resource_path&ref=' + str(ref) + '&extension=' + str(ext)
        
        response = self.do_request(query)


        if response.status_code == 200:
            download_url = response.text
        else:
            return None

        download_url = download_url.replace('\\/','/').strip('"')
        response = requests.get(download_url)
        
        if response.status_code == 200:
            return response.content
        else:
            return None

    

    #colid: the id of the collection you want to get the resource ids from
    #returns an array of resource descriptions as python objects
    def get_collection_resources(self, colid):
        search_term = '!collection' + str(colid)
        
        return self.do_search(search_term)



