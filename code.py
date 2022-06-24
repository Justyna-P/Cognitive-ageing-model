
# import simpy
import random
import pandas as pd
import random
import numpy as np

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt



#preferences = pd.read_excel("Preferences_random.xlsx")

#preferences = pd.read_excel("Preferences_random.xlsx")
preferences = pd.read_excel("Preferences_experiment.xlsx")

#prod_set = pd.read_excel("Products_pralki_original_skew.xlsx")  #weights
# products_pralki_skew_o.xlsx
# products_pralki_skew_m.xlsx
# products_pralki_bell.xlsx
prod_set = pd.read_excel("Products_pralki_modified_skew2.xlsx") 

#prod_set = pd.read_excel("Products_pralki_modified_bez0.xlsx")  #weights
brand_set = pd.read_excel("brands0.xlsx") 
bias_set = pd.read_excel("Bias_data.xlsx") 
review_set = pd.read_excel("reviews0.xlsx")

h_review_grid = pd.read_excel("h_review.xlsx")
h_brand_grid = pd.read_excel("h_brand.xlsx")
h_review_grid=h_review_grid.iloc[:,1:]
h_brand_grid=h_brand_grid.iloc[:,1:]




Agent_ID = preferences.iloc[1,:][0]
Agent_pref = preferences.iloc[1,:][1:]
Prod1 = prod_set.iloc[3,:]
Prod2 = prod_set.iloc[1,:]
included = [0,1,2,3,4,5,6]
included_l = [0,3,4,6]
included_l_2 = list(bias_set.iloc[29][7])
included_l_2 = list(map(int, included_l_2))

#if 1 in included:
 #   print("yes")


def shuffle_products(list_to_shuffle):
    pick = random.choice(list_to_shuffle)
    list_to_shuffle.remove(pick)
    return(pick)


def smooth_user_preference(x):
    return math.log(1+x, 2)

def utility_weighted(item_attributes, preferences_weights, included_list):
    utility = float(0)
    for i in range (0, len(item_attributes)):
        #print(i)
        if i in included_list:
            #print(str(1) + "yes")
            utility +=  item_attributes[i] * preferences_weights[i]
    return (utility)


def compare_weighted(item_attributes1, item_attributes2, preferences_weights, included_list, **kwargs):
    bias_r_state = kwargs.get('bias_r', 0)
    bias_b_state= kwargs.get('bias_b', 0)
    fav_b= kwargs.get('fav_b', 0)
    bias_r_prod1 = kwargs.get('prod_r1', 0)
    bias_r_prod2 = kwargs.get('prod_r2', 0)
    bias_b_prod1= kwargs.get('prod_b1', 0)   
    bias_b_prod2= kwargs.get('prod_b2', 0)   
    if fav_b==bias_b_prod1:
        bias_b_prod1=1
    else:
        bias_b_prod1=0       
    if fav_b==bias_b_prod2:
        bias_b_prod2=1
    else:
        bias_b_prod2=0     
       
    U1_bias = bias_r_state* bias_r_prod1*-200 + bias_b_state*bias_b_prod1*100
    U2_bias = bias_r_state* bias_r_prod2*-200 + bias_b_state*bias_b_prod2*100

    U1 = utility_weighted(item_attributes1, preferences_weights, included_list) + U1_bias
    U2 = utility_weighted(item_attributes2, preferences_weights, included_list) + U2_bias
    # print(U1_bias)
    # print(U2_bias)
    if U1>U2:
        return(1)
    else:
        return(2)
        
   
def compare_tally(item_attributes1, item_attributes2, preferences_weights, included_list, **kwargs):
    bias_r_state = kwargs.get('bias_r', 0)
    bias_b_state= kwargs.get('bias_b', 0)
    fav_b= kwargs.get('fav_b', 0)
    bias_r_prod1 = kwargs.get('prod_r1', 0)
    bias_r_prod2 = kwargs.get('prod_r2', 0)
    bias_b_prod1= kwargs.get('prod_b1', 0)   
    bias_b_prod2= kwargs.get('prod_b2', 0)   
    if fav_b==bias_b_prod1:
        bias_b_prod1=1
    else:
        bias_b_prod1=0       
    if fav_b==bias_b_prod2:
        bias_b_prod2=1
    else:
        bias_b_prod2=0     
       
    U1_bias = bias_r_state* bias_r_prod1*-200 + bias_b_state*bias_b_prod1*100
    U2_bias = bias_r_state* bias_r_prod2*-200 + bias_b_state*bias_b_prod2*100
    
    U1 = float(0) + U1_bias
    U2 = float(0) + U2_bias
    for i in range (0, len(item_attributes1)):
        if i in included_list:
            if item_attributes1[i]>item_attributes2[i]:
                U1 +=1
            elif item_attributes1[i]<item_attributes2[i]:
                U2 +=1
    #print(U1)
    #print(U2)
    if U1<U2:
        return(2)
    else:
        return(1)
                  
def compare_ttb(item_attributes1, item_attributes2, preferences_weights, included_list, **kwargs):
    bias_r_state = kwargs.get('bias_r', 0)
    bias_b_state= kwargs.get('bias_b', 0)
    fav_b= kwargs.get('fav_b', 0)
    bias_r_prod1 = kwargs.get('prod_r1', 0)
    bias_r_prod2 = kwargs.get('prod_r2', 0)
    bias_b_prod1= kwargs.get('prod_b1', 0)   
    bias_b_prod2= kwargs.get('prod_b2', 0)   
    if fav_b==bias_b_prod1:
        bias_b_prod1=1
    else:
        bias_b_prod1=0       
    if fav_b==bias_b_prod2:
        bias_b_prod2=1
    else:
        bias_b_prod2=0     
     
    U1_bias = bias_r_state* bias_r_prod1*-200 + bias_b_state*bias_b_prod1*100
    U2_bias = bias_r_state* bias_r_prod2*-200 + bias_b_state*bias_b_prod2*100
    
    if U1_bias>U2_bias:
        return(1)
    elif U2_bias>U1_bias:
        return(2)
    
    if item_attributes1.equals(item_attributes2):
         return(1)
    highest_w = max(preferences_weights, key=abs)
    while highest_w!=0:
      # print ("highest_w = " + str(highest_w))
      for i in range (0, len(item_attributes1)):
         # print(i) 
         if i in included_list:
             if preferences_weights[i]==highest_w:
                 # print("equal")
                 if item_attributes1[i]>item_attributes2[i]:
                     # print("jeden")
                     return(1)
                     break
                 elif item_attributes1[i]<item_attributes2[i]:
                     # print("dwa")
                     return(2)
                     break
      highest_w = highest_w - 1
    return(1)

def strategy_selection(age):
    random_n = random.randrange(100)
    if age =='O':
        if random_n<10:
            return('TTB')
        elif random_n < 40:
            return('TALLY')
        else:
            return('WADD')
    elif age == 'Y':
        if random_n<2:
            return('TTB')
        elif random_n < 8:
            return('TALLY')
        else:
            return('WADD')
        
        
def reduce_prod_set(included_list, prod_set_input, quantile, count_min):
    x=prod_set.quantile(quantile)
    for i in included_list:
        #print(i)
        #print(x[i])
        prod_set_output = prod_set_input[prod_set_input["A"+str(i)] >= x[i]]
        #print(prod_set_output.shape[0])
        if prod_set_output.shape[0]>=count_min:
            prod_set_input=prod_set_output
        #print(prod_set_input.shape[0])
    return(prod_set_input)

              

# products_pralki_skew_o.xlsx
# products_pralki_skew_m.xlsx
# products_pralki_bell.xlsx

#preferences = pd.read_excel("Preferences_random.xlsx")
preferences = pd.read_excel("Preferences_experiment.xlsx")

prod_set = pd.read_excel("products_pralki_bell_random_final.xlsx")  #weights


# 'Optymalna uzytecznosc per uzytkownik                
dane=pd.DataFrame()
for i in range (0, len(preferences)):
    atrybuty_names = []
    atrybuty_v = []
    Agent_ID = preferences.iloc[i,:][0]
    Agent_pref = preferences.iloc[i,:][1:]
    atrybuty_names.append('ID')
    atrybuty_v.append(Agent_ID)
    for j in range (0, len(prod_set)):
        Prod = prod_set.iloc[j,:] 
        util = utility_weighted(Prod, Agent_pref, included)
        atrybuty_names.append(str(j))
        atrybuty_v.append(util)
    df2 = pd.DataFrame([atrybuty_v], columns=atrybuty_names)
    dane = dane.append(df2)   
    
dane.to_excel('optimal_products_pralki_bell_random_final.xlsx')    

dane2 = dane
dane2 = dane2.set_index('ID')
utility_max = dane2.max(axis=1)
utility_min = dane2.min(axis=1)




dane_symulacja5=pd.DataFrame()
for modif in range (0,1):
    for heur_i in range (4,5):
    #print (h_brand_grid.iloc[0,:][heur_i])
        for k in range (0, 100):
            print("iteration:" + str(k))
            RANDOM_SEED = random.randrange(1000,2000,10) 
            # dane_symulacja2=pd.DataFrame()
            for i in range (0, len(preferences)):
                
                atrybuty_names = []
                atrybuty_v = []
                atrybuty_v.append(k)
                atrybuty_names.append('Simulation_no')
                Agent_ID = bias_set.iloc[i,:][0]
                Agent_AgeG = bias_set.iloc[i,:][4]
                Agent_pref = preferences.iloc[i,:][1:]
                #chosen 
                Agent_bias_r = h_review_grid.iloc[i,:][heur_i]
                Agent_bias_b = h_brand_grid.iloc[i,:][heur_i]
                # podmienic 1 i 2
                Agent_fav_b = bias_set.iloc[i,:][3]
                chosen_pref_list = list(bias_set.iloc[i,:][7])
                chosen_pref_list = list(map(int, chosen_pref_list))
                atrybuty_names.append('Reduction')
                if modif ==0:
                    prod_set_clone = reduce_prod_set(included_l, prod_set, 0.15, 30) 
                    atrybuty_v.append("15")
                if modif ==1:
                    prod_set_clone = reduce_prod_set(included_l, prod_set, 0.0001, 30)
                    atrybuty_v.append("none")
                product_indices = list(prod_set_clone.index)
                #product_indices = [*range(0, len(prod_set), 1)] 
                working_mem_capacity =  bias_set.iloc[i,:][5]
                atrybuty_names.append('ID')
                atrybuty_v.append(Agent_ID)
                atrybuty_names.append('AgeG')
                atrybuty_v.append(Agent_AgeG)
                atrybuty_names.append('bias_level')
                atrybuty_v.append(heur_i)
                atrybuty_names.append('bias_reviews')
                atrybuty_v.append(Agent_bias_r)
                atrybuty_names.append('bias_brand')
                atrybuty_v.append(Agent_bias_b)
                atrybuty_names.append('favourite_brand')
                atrybuty_v.append(Agent_fav_b)
                atrybuty_names.append('WM')
                atrybuty_v.append(working_mem_capacity)
                atrybuty_names.append('chosen_pref_list')
                atrybuty_v.append(chosen_pref_list)
                atrybuty_names.append('Age')
                if Agent_AgeG <3:
                    atrybuty_v.append('Y')
                else:
                    atrybuty_v.append('O')
                util_current = 0
                p = shuffle_products(product_indices)
                Prod_current = prod_set.iloc[p,:] 
                Prod_brand_current = int(brand_set.iloc[p])
                Prod_review_current = int(review_set.iloc[p])
                util = utility_weighted(Prod_current, Agent_pref, included)
                strat = 'WADD'
                atrybuty_names.append("chosen_prod_" + str(0))
                atrybuty_v.append(p)
                atrybuty_names.append("real_util_" + str(0))
                atrybuty_v.append((util-utility_min.iloc[i])/(utility_max.iloc[i]-utility_min.iloc[i]))
               
                for j in range (1, 50):
                    p = shuffle_products(product_indices)
                    Prod_new = prod_set.iloc[p,:]
                    Prod_brand_new = int(brand_set.iloc[p])
                    Prod_review_new = int(review_set.iloc[p])
                    if Agent_AgeG <3:
                        strat = strategy_selection('Y')
                    else:
                        strat = strategy_selection('O')
                    choice = 1
                    if strat == 'TTB':
                        # print('TTB')
                        choice = compare_ttb(Prod_current, Prod_new, Agent_pref, chosen_pref_list, bias_r=Agent_bias_r, 
                                             prod_r1= Prod_review_current, prod_r2 =  Prod_review_new, bias_b=Agent_bias_b, fav_b = Agent_fav_b, 
                                             prod_b1= Prod_brand_current, prod_b2 = Prod_brand_new) 
                        # print(choice)
                    elif strat == 'TALLY':
                        choice = compare_tally(Prod_current, Prod_new, Agent_pref, chosen_pref_list, bias_r=Agent_bias_r, 
                                             prod_r1= Prod_review_current, prod_r2 =  Prod_review_new, bias_b=Agent_bias_b, fav_b = Agent_fav_b, 
                                             prod_b1= Prod_brand_current, prod_b2 = Prod_brand_new)  
                    else:
                        choice = compare_weighted(Prod_current, Prod_new, Agent_pref, chosen_pref_list, bias_r=Agent_bias_r, 
                                             prod_r1= Prod_review_current, prod_r2 =  Prod_review_new, bias_b=Agent_bias_b, fav_b = Agent_fav_b, 
                                             prod_b1= Prod_brand_current, prod_b2 = Prod_brand_new) 
                  
                    
                    if choice ==2:
                        Prod_current=Prod_new
                        Prod_brand_current = Prod_brand_new
                        Prod_review_current = Prod_review_new
                    util_new = utility_weighted(Prod_current, Agent_pref, included)
            
                    atrybuty_names.append("chosen_prod_" + str(j))
                    atrybuty_v.append(p)
                    # atrybuty_names.append("strategy_" + str(j))
                    # atrybuty_v.append(strat)           
                    atrybuty_names.append("choice_" + str(j))
                    atrybuty_v.append(choice)
                    atrybuty_names.append("real_util_" + str(j))
                    atrybuty_v.append((util_new-utility_min.iloc[i])/(utility_max.iloc[i]-utility_min.iloc[i]))
                df4 = pd.DataFrame([atrybuty_v], columns=atrybuty_names)
                dane_symulacja5 = dane_symulacja5.append(df4)
        dane_symulacja5.to_excel("Wyniki_bell.xlsx")
dane_symulacja_part1 = dane_symulacja5[267301:]
dane_symulacja_part1.to_excel("Wyniki_modifiedskew_no_reduction.xlsx")




set_to_ignore = get_items_interacted(1, interactions_full_indexed_df)



# 'Symulacja6 - z systemem rekomendacyjnym

def shuffle_products(list_to_shuffle):
    pick = random.choice(list_to_shuffle)
    list_to_shuffle.remove(pick)
    return(pick)
    
def smooth_user_preference(x):
    return math.log(1+x, 2)

def get_items_interacted(Client, interactions_df):
    interacted_items = interactions_df.loc[Client]['Product']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, Client, items_to_ignore=[], topn=100, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['Product'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'Product', 
                                                          right_on = 'Product')#[['eventStrength', 'Product', 'Brand', 'Ratings']]


        return recommendations_df



class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, Client, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[Client].sort_values(ascending=False) \
                                    .reset_index().rename(columns={Client: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['Product'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'Product', 
                                                          right_on = 'Product')[['recStrength', 'Product', 'title', 'url', 'lang']]


        return recommendations_df



# Znalezienie bliźniaka preferencji
preferences_values = preferences.drop(columns=['A_ID'])
cosine_similarities_pref = cosine_similarity(preferences_values, preferences_values)
index_list=bias_set.loc[bias_set['Age_group'] == 3].index #selectin old to drop
cosine_similarities_reduced=pd.DataFrame(cosine_similarities_pref).drop(columns=list(index_list))

similarity_twin=[]
for i in range (0, len(preferences)):
    all_similarities = cosine_similarities_reduced.iloc[i,:]
    try:
        all_similarities = all_similarities.drop(i)
    except:
        pass
    similarity_twin.append(all_similarities.idxmax())



# Znalezienie competence based
preferences_values = preferences.drop(columns=['A_ID'])
cosine_similarities_pref = cosine_similarity(preferences_values, preferences_values)
index_list=bias_set.loc[bias_set['WM'] < 5].index #selectin low memory to drop
cosine_similarities_reduced=pd.DataFrame(cosine_similarities_pref).drop(columns=list(index_list))

competence_based=[]
for i in range (0, len(preferences)):
    all_similarities = cosine_similarities_reduced.iloc[i,:]
    try:
        all_similarities = all_similarities.drop(i)
    except:
        pass
    competence_based.append(all_similarities.idxmax())



#p = cf_recommender_model.recommend_items(1, items_to_ignore=set_to_ignore, topn=15, verbose=False).iloc[0,0]
rec_sys_params = (5, 10, 30)
sets_names =("15p_", "none_")
tran=4
dane_symulacja6=pd.DataFrame()
for sets in range(0,1):
    set_name = sets_names[sets]
    print(set_name)
    for param in range(1):
        #print(param)
        top_n_param_i = rec_sys_params[param]
        #print(top_n_param_i)
        for tran in range(0,10):
            products = prod_set #pd.read_excel(prod_sets[prod])
            items_df = products
            items_df = items_df.reset_index()
            items_df = items_df.rename(columns = {'index':'Product'})
            
            ds2 =products#.drop(columns=['ID'])
            cosine_similarities = cosine_similarity(ds2, ds2)
            similaritems = {}
            for idx in range(ds2.shape[0]):
                similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
                similaritems[idx] = similar_indices[1:]
                
            interactions_df = pd.read_excel('transactions'+str(set_name)+str(tran)+'.xlsx') 
            interactions_df['eventStrength'] = 1
            interactions_full_df = interactions_df \
                    .groupby(['Client', 'Product'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
            interactions_full_indexed_df = interactions_full_df.set_index('Client')
            item_popularity_df = interactions_full_df.groupby('Product')['eventStrength'].sum().sort_values(ascending=False).reset_index()
            popularity_model = PopularityRecommender(item_popularity_df, items_df)
            
            users_items_pivot_matrix_df = interactions_full_df.pivot(index='Client', 
                                                          columns='Product', 
                                                          values='eventStrength').fillna(0)
            #users_items_pivot_matrix_df.head(10)
            users_items_pivot_matrix = users_items_pivot_matrix_df.values
            users_ids = list(users_items_pivot_matrix_df.index)
            NUMBER_OF_FACTORS_MF = top_n_param_i
            U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
            sigma = np.diag(sigma)
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
            cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
            cf_recommender_model = CFRecommender(cf_preds_df, items_df)          
            register = pd.DataFrame(columns=['Client','Product', 'RecSys', 'Utility_perceived','Utility_real'])
           #i=5
            for k in range (0, 10):
                print("param:"+ str(top_n_param_i) +" tran_set:" + str(tran)+"_iteration:" + str(k))
                PRODUCTS_V_LIST = list(range(0, products.shape[0]))
                RANDOM_SEED = random.randrange(1000,2000,10) 
                # dane_symulacja2=pd.DataFrame()
                for i in range (1, len(preferences)): #dla każdego użytkownika i:
          #      for i in range (1, 6): #dla każdego użytkownika i:
                    nID=i-1
                    PRODUCTS_V_LIST = list(range(0, products.shape[0]))
                    set_to_ignore = get_items_interacted(i, interactions_full_indexed_df)
                    items_purchased = pd.DataFrame(list(set_to_ignore))
                    items_purchased =items_purchased.set_index(0)             
                    items_purchased = list(set_to_ignore)
                    pref_table = pd.DataFrame()
                    for item in range(len(items_purchased)):
                        s_items = pd.DataFrame(similaritems[items_purchased[item]][:top_n_param_i])
                        pref_table = pref_table.append(s_items)
                    pref_table = pref_table[0].value_counts()
                    pref_table = pref_table.reset_index().drop(columns=[0])
                    pref_table = pref_table[~pref_table['index'].isin(set_to_ignore)] 
                    
                    twin_index = similarity_twin[nID]+1
                    set_to_recommend_twin = get_items_interacted(twin_index, interactions_full_indexed_df)
                    competence_index = competence_based[nID]+1
                    set_to_recommend_competence = get_items_interacted(competence_index, interactions_full_indexed_df)
                    
                    
                    #print("i="+str(i))
                    #print(pref_table)
                    nID=i-1
                    product_indices = [*range(0, len(prod_set), 1)] 
                    atrybuty_names = []
                    atrybuty_v = []
                    atrybuty_v.append(k)
                    atrybuty_names.append('Simulation_no')
                    atrybuty_v.append(tran)
                    atrybuty_names.append('reduction_param')
                    atrybuty_v.append(set_name)
                    atrybuty_names.append('Training_set')           #i=1
                    Agent_ID = bias_set.iloc[nID,:][0]
                    Agent_AgeG = bias_set.iloc[nID,:][4]
                    Agent_pref = preferences.iloc[nID,:][1:]
                    #Agent_bias_r = bias_set.iloc[nID,:][1]
                    #Agent_bias_b = bias_set.iloc[nID,:][2]
                    #chosen 
                    Agent_bias_r = h_review_grid.iloc[nID,:][5]
                    Agent_bias_b = h_brand_grid.iloc[nID,:][5]
                    
                    Agent_fav_b = bias_set.iloc[nID,:][3]
                    chosen_pref_list = list(bias_set.iloc[nID,:][7])
                    chosen_pref_list = list(map(int, chosen_pref_list))
                    working_mem_capacity =  bias_set.iloc[nID,:][5]
                    atrybuty_names.append('ID')
                    atrybuty_v.append(Agent_ID)
                    #print("ID="+str(Agent_ID))
                    
                    
        
                    atrybuty_names.append('AgeG')
                    atrybuty_v.append(Agent_AgeG)
                    atrybuty_names.append('bias_reviews')
                    atrybuty_v.append(Agent_bias_r)
                    atrybuty_names.append('bias_brand')
                    atrybuty_v.append(Agent_bias_b)
                    atrybuty_names.append('favourite_brand')
                    atrybuty_v.append(Agent_fav_b)
                    atrybuty_names.append('WM')
                    atrybuty_v.append(working_mem_capacity)
                    atrybuty_names.append('chosen_pref_list')
                    atrybuty_v.append(chosen_pref_list)
                    atrybuty_names.append('Age')
                    if Agent_AgeG <3:
                        atrybuty_v.append('Y')
                    else:
                        atrybuty_v.append('O')
                    atrybuty_names.append('Recommender_params')
                    atrybuty_v.append(top_n_param_i)
                    util_current = 0
                    for r in range(1,6):
                        #print(r)
                        if r ==0:
                            p = shuffle_products(PRODUCTS_V_LIST)
                        elif r ==1:
                            p = popularity_model.recommend_items(i, items_to_ignore=set_to_ignore, topn=15, verbose=False).reset_index(drop=True).iloc[0,0]
                        elif r ==2:
                            p = cf_recommender_model.recommend_items(i, items_to_ignore=set_to_ignore, topn=30, verbose=False).iloc[0,0]
                        elif r==4:
                            p = list(set_to_recommend_twin)[0]
                        elif r==5:
                            p = list(set_to_recommend_competence)[0]
                            #nowy algorytm
                        else:
                            p = pref_table.iloc[0,0]
                        #p = shuffle_products(product_indices)
                        Prod_current = prod_set.iloc[p,:] 
                        Prod_brand_current = int(brand_set.iloc[p])
                        Prod_review_current = int(review_set.iloc[p])
                        util = utility_weighted(Prod_current, Agent_pref, included)
                        strat = 'WADD'
                        atrybuty_names.append("chosen_prod_" + str(0))
                        atrybuty_v.append(p)
                        atrybuty_names.append("real_util_" + str(0))
                        atrybuty_v.append((util-utility_min.iloc[nID])/(utility_max.iloc[nID]-utility_min.iloc[nID]))
                       
                        for j in range (1, 20): #przejrzyj losowo wybrane 30 produktów
                            p = shuffle_products(product_indices) #p to wybrany produkt
                            
                            if r ==0:
                                p = shuffle_products(PRODUCTS_V_LIST)
                            elif r ==1:
                                p = popularity_model.recommend_items(i, items_to_ignore=set_to_ignore, topn=35, verbose=False).reset_index(drop=True).iloc[j,0]
                            elif r ==2:
                                p = cf_recommender_model.recommend_items(i, items_to_ignore=set_to_ignore, topn=30, verbose=False).reset_index(drop=True).iloc[j,0]
                            
                            elif r==4:
                                try:
                                    p = list(set_to_recommend_twin)[j]
                                except:
                                    p = list(set_to_recommend_twin)[len(set_to_recommend_twin)-1]
                            elif r==5:
                                try:
                                    p = list(set_to_recommend_competence)[j]
                                except:
                                    p = list(set_to_recommend_competence)[len(set_to_recommend_competence)-1]
                            else:
                                try:
                                    p = pref_table.iloc[j,0]
                                except:
                                    p= int(pref_table.iloc[(pref_table.count()-1),0])
                            
                            Prod_new = prod_set.iloc[p,:]
                            Prod_brand_new = int(brand_set.iloc[p])
                            Prod_review_new = int(review_set.iloc[p])
                            if Agent_AgeG <3:
                                strat = strategy_selection('Y')
                            else:
                                strat = strategy_selection('O')
                            choice = 1
                            if strat == 'TTB':
                                # print('TTB')
                                choice = compare_ttb(Prod_current, Prod_new, Agent_pref, chosen_pref_list, bias_r=Agent_bias_r, 
                                                     prod_r1= Prod_review_current, prod_r2 =  Prod_review_new, bias_b=Agent_bias_b, fav_b = Agent_fav_b, 
                                                     prod_b1= Prod_brand_current, prod_b2 = Prod_brand_new) 
                                # print(choice)
                            elif strat == 'TALLY':
                                choice = compare_tally(Prod_current, Prod_new, Agent_pref, chosen_pref_list, bias_r=Agent_bias_r, 
                                                     prod_r1= Prod_review_current, prod_r2 =  Prod_review_new, bias_b=Agent_bias_b, fav_b = Agent_fav_b, 
                                                     prod_b1= Prod_brand_current, prod_b2 = Prod_brand_new)  
                            else:
                                choice = compare_weighted(Prod_current, Prod_new, Agent_pref, chosen_pref_list, bias_r=Agent_bias_r, 
                                                     prod_r1= Prod_review_current, prod_r2 =  Prod_review_new, bias_b=Agent_bias_b, fav_b = Agent_fav_b, 
                                                     prod_b1= Prod_brand_current, prod_b2 = Prod_brand_new)      
                            
                            if choice ==2:
                                Prod_current=Prod_new
                                Prod_brand_current = Prod_brand_new
                                Prod_review_current = Prod_review_new
                                
                            util_new = utility_weighted(Prod_current, Agent_pref, included)
                    
                            #atrybuty_names.append("chosen_prod_" + str(j)+"_"+str(r))
                            #atrybuty_v.append(p)
                            # atrybuty_names.append("recommender_system_" + str(j))
                            # atrybuty_v.append(r)
                            atrybuty_names.append("real_util_" + str(j)+"_recsys:"+str(r))
                            atrybuty_v.append((util_new-utility_min.iloc[nID])/(utility_max.iloc[nID]-utility_min.iloc[nID]))
                    df4 = pd.DataFrame([atrybuty_v], columns=atrybuty_names)
                    dane_symulacja6 = dane_symulacja6.append(df4)
dane_symulacja6.to_excel("Wyniki_after_recommender_bell.xlsx")













