from app2 import app

##First Positive Test case for '/hello' route 
def test_hello_route_success():
    tester = app.test_client()
    response = tester.get('/hello')

    assert response.status_code == 200

# #Creating Failure Test case :
# def test_hello_route_failure():
#     tester = app.test_client()
#     response = tester.get('/hello')

#     assert response.status_code == 500

# # Positive Test Case for '/predict' route : 
# def test_predict_route_success(): 
#     tester  = app.test_client()
 
#     data = { "gestation":[279],
#                      "parity":[0],
#                      "age": [27],
#                      "height": [70],
#                      "weight": [100],
#                      "smoke": [1]
#                         }

#     response = tester.post("/predict",json = data)

#     assert response.status_code == 200    

# Negative Test Case for '/predict' route : 
def test_predict_route_Invalid_Data(): 
    tester  = app.test_client()
 

    response = tester.post("/predict",json = {})

    assert response.status_code == 400   


    # Negative Test Case for '/predict' route : 
def test_predict_route_wrong_url(): 
    tester  = app.test_client()
 
    data = { "gestation":[279],
                     "parity":[0],
                     "age": [27],
                     "height": [70],
                     "weight": [100],
                     "smoke": [1]
                        }

    response = tester.post("/oredict",json = data)

    assert response.status_code == 404     


    # Negative Test Case for '/predict' route : 
def test_predict_route_wrong_method(): 
    tester  = app.test_client()
 
    data = { "gestation":[279],
                     "parity":[0],
                     "age": [27],
                     "height": [70],
                     "weight": [100],
                     "smoke": [1]
                        }

    response = tester.get("/predict",json = data)

    assert response.status_code == 405    