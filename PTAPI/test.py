from algorithm import recommendation
from preprocess import read_data
from lxml import etree
from nltk.stem import SnowballStemmer
from algorithm import similarity
from nltk.tokenize import WordPunctTokenizer
import gensim
import cPickle as pickle
from bs4 import BeautifulSoup
import util
import time

import sys
reload(sys)
sys.setdefaultencoding('utf8')


w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed') # pre-trained word embedding
idf = pickle.load(open('../data/idf', 'rb')) # pre-trained idf value of all words in the w2v dictionary
questions = pickle.load(open('../data/api_questions_pickle_new', 'rb')) # the pre-trained knowledge base of api-related questions (about 120K questions)
questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = pickle.load(open('../data/parent', 'rb')) # parent is a dict(), which stores the ids of each query's duplicate questions

querys = read_data.read_querys_from_file()
#querys = querys[:10]

print 'loading data finished'

jisu=0
mrr = 0.0
map = 0.0
tot = 0

pre10 = 0.0
pre1 = 0.0
pre3=0.0
count=0
for item in querys:

    #query = item[0].title
    query = item[0]
    true_apis = item[1]

    query_words = WordPunctTokenizer().tokenize(query.lower())
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)

    #top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
    #recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)
    top_questions_title = recommendation.get_topk_questions_title(query, query_matrix, query_idf_vector, questions, 10,parent)
    user_selest = list()
    i = 0
    while i < 3:
        user_selest.append(top_questions_title[i][1])
        i = i + 1
    print (query)


    select_1 = user_selest[0][0] + ' ' + query
    #select_1 = query + ' ' + user_selest[0][0]
    select_1_words = WordPunctTokenizer().tokenize(select_1.lower())
    if select_1_words[-1] == '?':
        select_1_words = select_1_words[: -1]
    select_1_words = [SnowballStemmer('english').stem(word) for word in select_1_words]
    select_1_matrix = similarity.init_doc_matrix(select_1_words, w2v)
    select_1_idf_vector = similarity.init_doc_idf_vector(select_1_words, idf)
    top_questions = recommendation.get_topk_questions(select_1, select_1_matrix, select_1_idf_vector, questions, 5 ,parent)
    recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions,javadoc, javadoc_dict_methods, -1)

    print (recommended_api)
    pos = -1
    tmp_map = 0.0
    hits = 0.0
    count_1 = 0
    for i, api in enumerate(recommended_api):
        if api in true_apis and pos == -1:
            pos = i + 1
        if api in true_apis:
            hits += 1
            tmp_map += hits / (i + 1)
    tmp_map /= len(true_apis)
    tmp_mrr = 0.0
    if pos != -1:
        tmp_mrr = 1.0 / pos

    if pos < 11 | pos != -1:
        count_10 = 1
    else:
        count_10 = 0

    if pos == 1:
        count_1 = 1
    else:
        count_1 = 0

    if pos < 4 | pos != -1:
        count_3 = 1
    else:
        count_3 = 0

    pre3 += count_3
    pre10 += count_10
    pre1 += count_1
    map += tmp_map
    mrr += tmp_mrr

    print pre1
    print pre3
    print pre10
    print 'this question mrr:', tmp_mrr, 'this question map:', tmp_map, 'api Position:', pos, query, true_apis, len(recommended_api)
    jisu = jisu + 1
    print(jisu)




print pre10, pre1, pre3
print 'precision@3', pre3 / len(querys)
print 'precision@1:', pre1 / len(querys)
print 'precision@10:', pre10 / len(querys)
print 'mrr:', mrr / len(querys), 'quersyslen', len(querys)
print 'map', map / len(querys)

    # for i, api in enumerate(recommended_api):
    #     if i==10:
    #         break
    #     print api,'rank',i
    #     recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)





# import random
#
# from algorithm import recommendation
# from preprocess import read_data
# from lxml import etree
# from nltk.stem import SnowballStemmer
# from algorithm import similarity
# from nltk.tokenize import WordPunctTokenizer
# import gensim
# import cPickle as pickle
# from bs4 import BeautifulSoup
# import util
# import time
# import random
#
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
#
#
# w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed') # pre-trained word embedding
# idf = pickle.load(open('../data/idf', 'rb')) # pre-trained idf value of all words in the w2v dictionary
# questions = pickle.load(open('../data/api_questions_pickle_new', 'rb')) # the pre-trained knowledge base of api-related questions (about 120K questions)
# questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
# javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
# javadoc_dict_classes = dict()
# javadoc_dict_methods = dict()
# recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
# parent = pickle.load(open('../data/parent', 'rb')) # parent is a dict(), which stores the ids of each query's duplicate questions
#
# querys = read_data.read_querys_from_file()
# #querys = querys[:10]
#
# print 'loading data finished'
#
# jisu=0
# mrr = 0.0
# map = 0.0
# tot = 0
#
# pre10 = 0.0
# pre1 = 0.0
# pre3=0.0
# count=0
#
#
# # for item in querys:
# #
# #     #query = item[0].title
# #     query = item[0]
# #     true_apis = item[1]
# #
# #     query_words = WordPunctTokenizer().tokenize(query.lower())
# #     query_words = [SnowballStemmer('english').stem(word) for word in query_words]
# #
# #     query_matrix = similarity.init_doc_matrix(query_words, w2v)
# #     query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
# #
# #     #top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
# #     #recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)
# #     top_questions_title = recommendation.get_topk_questions_title(query, query_matrix, query_idf_vector, questions, 10, parent)
# #     user_selest = list()
# #
# #     i = 0
# #     #select_1 = user_selest[0][0]
# #     while i < 10:
# #         user_selest.append(top_questions_title[i][1])
# #         i = i + 1
# #         #select_1 = select_1 + '' + query
# #     print (query)
# #
# #
# #     k = random.randint(1, 10)
# #     j = random.randint(1, 10)
# #     if j != k:
# #         select_1 = user_selest[0][0] + ' ' + user_selest[0][j] + ' ' + user_selest[0][k] + ' ' + query
# #     #select_1 = query + '' + user_selest[0][0]
# #     select_1_words = WordPunctTokenizer().tokenize(select_1.lower())
# #     if select_1_words[-1] == '?':
# #         select_1_words = select_1_words[: -1]
# #     select_1_words = [SnowballStemmer('english').stem(word) for word in select_1_words]
# #     select_1_matrix = similarity.init_doc_matrix(select_1_words, w2v)
# #     select_1_idf_vector = similarity.init_doc_idf_vector(select_1_words, idf)
# #     top_questions = recommendation.get_topk_questions(select_1, select_1_matrix, select_1_idf_vector, questions, 50 ,parent)
# #     recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions,javadoc, javadoc_dict_methods, -1)
# #
# #     print (recommended_api)
# #     pos = -1
# #     tmp_map = 0.0
# #     hits = 0.0
# #     count_1 = 0
# #     for i, api in enumerate(recommended_api):
# #         if api in true_apis and pos == -1:
# #             pos = i + 1
# #         if api in true_apis:
# #             hits += 1
# #             tmp_map += hits / (i + 1)
# #     tmp_map /= len(true_apis)
# #     tmp_mrr = 0.0
# #     if pos != -1:
# #         tmp_mrr = 1.0 / pos
# #
# #     if pos < 11 | pos != -1:
# #         count_10 = 1
# #     else:
# #         count_10 = 0
# #
# #     if pos == 1:
# #         count_1 = 1
# #     else:
# #         count_1 = 0
# #
# #     if pos < 4 | pos != -1:
# #         count_3 = 1
# #     else:
# #         count_3 = 0
# #
# #     pre3 += count_3
# #     pre10 += count_10
# #     pre1 += count_1
# #     map += tmp_map
# #     mrr += tmp_mrr
# #
# #     print pre1
# #     print pre3
# #     print pre10
# #     print 'this questt pre3ion mrr:', tmp_mrr, 'this question map:', tmp_map, 'api Position:', pos, query, true_apis, len(recommended_api)
# #     jisu = jisu + 1
# #     print(jisu)
# #
# #
# #
# #
# # print pre10, pre1, pre3
# # print 'precision@3', pre3 / len(querys)
# # print 'precision@1:', pre1 / len(querys)
# # print 'precision@10:', pre10 / len(querys)
# # print 'mrr:', mrr / len(querys), 'quersyslen', len(querys)
# # print 'map', map / len(querys)
#
#     # for i, api in enumerate(recommended_api):
#     #     if i==10:
#     #         break
#     #     print api,'rank',i
#     #     recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)
#
#
# # h = 0
# # avg_mrr = 0.0
# # avg_map = 0.0
# #
# # avg_pre1 = 0.0
# # avg_pre3 = 0.0
# # avg_pre10 = 0.0
# # while h < 10:
# #     h += 1
# #     jisu = 0
# #     mrr = 0.0
# #     map = 0.0
# #     tot = 0
# #
# #     pre10 = 0.0
# #     pre1 = 0.0
# #     pre3 = 0.0
# #     count = 0
#
#
# for item in querys:
#
#     # query = item[0].title
#     query = item[0]
#     true_apis = item[1]
#
#     query_words = WordPunctTokenizer().tokenize(query.lower())
#     query_words = [SnowballStemmer('english').stem(word) for word in query_words]
#
#     query_matrix = similarity.init_doc_matrix(query_words, w2v)
#     query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
#
#     # top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
#     # recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)
#     top_questions_title = recommendation.get_topk_questions_title(query, query_matrix, query_idf_vector, questions,
#                                                                   3, parent)
#     user_selest = list()
#
#     i = 0
#     # select_1 = user_selest[0][0]
#     while i < 10:
#         user_selest.append(top_questions_title[i][1])
#         i = i + 1
#         # select_1 = select_1 + '' + query
#     print (query)
#
#     # k = random.randint(1, 10)
#     # j = random.randint(1, 10)
#     # if j != k:
#     #     select_1 = user_selest[0][0] + ' ' + user_selest[0][j] + ' ' + user_selest[0][k] + ' ' + query
#     select_1 = query + '' + user_selest[0][0]
#     select_1_words = WordPunctTokenizer().tokenize(select_1.lower())
#     if select_1_words[-1] == '?':
#         select_1_words = select_1_words[: -1]
#     select_1_words = [SnowballStemmer('english').stem(word) for word in select_1_words]
#     select_1_matrix = similarity.init_doc_matrix(select_1_words, w2v)
#     select_1_idf_vector = similarity.init_doc_idf_vector(select_1_words, idf)
#     top_questions = recommendation.get_topk_questions(select_1, select_1_matrix, select_1_idf_vector, questions, 50,
#                                                       parent)
#     recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions,
#                                                    javadoc, javadoc_dict_methods, -1)
#
#     print (recommended_api)
#     pos = -1
#     tmp_map = 0.0
#     hits = 0.0
#     count_1 = 0
#     for i, api in enumerate(recommended_api):
#         if api in true_apis and pos == -1:
#             pos = i + 1
#         if api in true_apis:
#             hits += 1
#             tmp_map += hits / (i + 1)
#     tmp_map /= len(true_apis)
#     tmp_mrr = 0.0
#     if pos != -1:
#         tmp_mrr = 1.0 / pos
#
#     if pos < 6 | pos != -1:
#         count_10 = 1
#     else:
#         count_10 = 0
#
#     if pos == 1:
#         count_1 = 1
#     else:
#         count_1 = 0
#
#     if pos < 4 | pos != -1:
#         count_3 = 1
#     else:
#         count_3 = 0
#
#     pre3 += count_3
#     pre10 += count_10
#     pre1 += count_1
#     map += tmp_map
#     mrr += tmp_mrr
#
#     print pre1
#     print pre3
#     print pre10
#     print 'this questt pre3ion mrr:', tmp_mrr, 'this question map:', tmp_map, 'api Position:', pos, query, true_apis, len(
#         recommended_api)
#     jisu = jisu + 1
#     print(jisu)
#
# # avg_pre1 += pre1
# # avg_pre3 += pre3
# # avg_pre10 += pre10
#
# # avg_map = avg_map + (map / len(querys))
# # avg_mrr = avg_mrr + (mrr / len(querys))
#
# print pre10, pre1, pre3
# print 'precision@3', pre3 / len(querys)
# print 'precision@1:', pre1 / len(querys)
# print 'precision@10:', pre10 / len(querys)
# print 'mrr:', mrr / len(querys), 'quersyslen', len(querys)
# print 'map', map / len(querys)
#
#     # for i, api in enumerate(recommended_api):
#     #     if i==10:
#     #         break
#     #     print api,'rank',i
#     #     recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)
#
# # print '---------------------------------------------------------------------------------------------------'
# # print avg_pre1/10, avg_pre3/10, avg_pre10/10
# # print avg_mrr /10
# # print avg_map/10
#
#
