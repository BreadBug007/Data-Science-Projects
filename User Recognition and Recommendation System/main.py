import recognize
import recommend

pred_face = recognize.make_prediction(model_name='face_enc')
recommend.recommend_products(pred_face)
