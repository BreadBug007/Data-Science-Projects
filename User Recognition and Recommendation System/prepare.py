import recognize
import recommend

# train dlib model with deep learning to recognize face
recognize.train_model(model_name="face_enc_live",
                      training_dataset="face_data/training")
# train word2vec model to recommend products
recommend.train_model(model_name="word2vec.model",
                      sales_data="sales_data/OnlineRetail.csv",
                      products_json="products.json")
