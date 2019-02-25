# article-popularity-predictor

A simple model-as-a-service flask app that predicts the popularity of an article on social media based on its title and publication time. 
The prediction is the log transformed and scaled number of views, e.g., an article which gets the mean number of views will 
result in a model prediction of 0.0, more than average would be a positive number, less than average a negative one

Service can be run locally by running the following commands from inside the git repo
```
pip install -r requirements.txt
flask run
```

Model endpoint
`http://127.0.0.1:5000/model`

Example valid POST request payload
```
{"timestamp":"2015-09-17 20:50:00", "description":"The Seattle Seahawks are a football team owned by Paul Allen."}
```

Example response
```
{"prediction": 0.3627112183586569}
```
