import requests
from flask import redirect, session

NCTU_OAUTH_URL = 'https://id.nctu.edu.tw'
NYCU_OAUTH_URL = 'https://id.nycu.edu.tw'


class Oauth(object):
    def __init__(self, redirect_uri, client_id, client_secret, oauth_url):
        self.grant_type = 'authorization_code'
        self.oauth_url = oauth_url
        self.code = None
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def authorize(self):
        get_code_url = self.oauth_url + '/o/authorize/?client_id=' + \
            self.client_id + '&scope=profile&response_type=code'
        return redirect(get_code_url)

    def get_token(self, code):
        self.code = code
        get_token_url = self.oauth_url + '/o/token/'
        data = {
            'grant_type': 'authorization_code',
            'code': self.code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri
        }
        access_token = requests.post(
            get_token_url, data=data).json().get('access_token', None)

        if access_token:
            session['nctu_token'] = access_token
            session['logged_in'] = True
            profile = self.get_profile()
            session['username'] = profile['username']
            return True

        return False

    def get_profile(self):
        token = session.get('nctu_token')
        headers = {
            'Authorization': 'Bearer ' + token
        }
        get_profile_url = self.oauth_url + '/api/profile/'

        data = requests.get(get_profile_url, headers=headers).json()

        return data
