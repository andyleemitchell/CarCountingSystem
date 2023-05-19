import github
import datetime

token = "__token__"  # https://github.com/settings/tokens to generate a token 

filename = 'carpark-spaces.json'
content = open(filename, 'r').read()

gh = github.Github(token)
# https://docs.github.com/en/rest/gists/gists?apiVersion=2022-11-28#update-a-gist
gist_id = "__gist-id__" 
gist = gh.get_gist(gist_id)

iso_date = datetime.datetime.now().isoformat()
description = "updated: " + iso_date

gist.edit(
    description=description,
    files={"ee399.json": github.InputFileContent(content=content)},
)
