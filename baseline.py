import numpy as np


class BaselineRating:
    def __init__(self,  function="RMSE"):
        self.function = function
        assert(self.function == "RMSE" or self.function == "MAE")
    
    
    def  fit(self, data):
        pass
    
    
    def predict(self, data):
        pass
    
#global average
class AvrRating(BaselineRating):  
    def fit(self, data):
        if self.function == "RMSE":
            self.const = np.mean(data[:, 2])
        if self.function == "MAE":
            self.const = np.median(data[:, 2])
    
    
    def predict(self, data):
        return self.const * np.ones(data.shape[0])
 
#average by user
#method think every user will be in fitting
class AvrUserRating(BaselineRating):
    def fit(self, data):
        avruser = np.empty(data[:, 0].max())
        users = data[:, 0]
        for i in range(avruser.shape[0]):
            user_data = data[data[:, 0] == i + 1, 2]
            if user_data.shape[0] > 0:
                if self.function == "RMSE":
                    avruser[i] = np.mean(user_data)
                elif self.function == "MAE":
                    avruser[i] = np.median(user_data)
        self.avruser = avruser
    
    
    def predict(self, data):
        return self.avruser[data[:, 0] - 1]

#average by item
#some item can absent in fitting
class AvrItemRating(BaselineRating):
    def fit(self, data):
        avritem = np.empty(data[:, 1].max())
        items = data[:, 1]
        for i in range(avritem.shape[0]):
            item_data = data[data[:, 1] == i + 1, 2]
            if item_data.shape[0] > 0:
                if self.function == "RMSE":
                    avritem[i] = np.mean(item_data)
                elif self.function == "MAE":
                    avritem[i] = np.median(item_data)
            else:
                avritem[i] = 0
        avritem[avritem == 0] = np.mean(avritem)
        self.avritem = avritem
        
        
    def predict(self, data):
        predict_max_item = data[:, 1].max()
        loc_avritem = self.avritem
        diff = predict_max_item - self.avritem.shape[0]
        if diff > 0:
            loc_avritem = np.concatenate((loc_avritem, 
                loc_avritem.mean() * np.ones(diff)))
        return loc_avritem[data[:, 1] - 1]
                
#full random from min to max      
class RandomRating(BaselineRating):
    def fit(self, data):
        self.min = data[:, 2].min()
        self.max = data[:, 2].max()
    
    
    def predict(self, data):
        return np.random.randint(self.min, self.max + 1, size=data.shape[0])
    
    