[Click Here to take you to User Recommandation Engine app](https://user-recommendation-engine.herokuapp.com/)


## Steps to run on local

``` 
pip install pipenv
pipenv install flask gunicorn 
pipenv install -r requirements.txt
pipenv shell
python app.py
```
## Steps to run on heroku

```
git init 
git add .
git commit -m "Initial Commit"
heroku login
heroku create user-recommendation-engine
git push heroku master
```

## Important files to have before deploying on heroku
- Procfile
- Pipfile
- runtime.txt
