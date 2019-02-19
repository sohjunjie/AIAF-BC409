from lstm_doc2vec import LSTMdoc2vec

d = LSTMdoc2vec()
m = d.create_model()
x1, x2, y1, y2 = d._get_train_dataseq()
vx1, vx2, vy1, vy2  = d._get_dataseq(d.valid_start_idx_list)
ex1, ex2, ey1, ey2 = d._get_dataseq(d.evalu_start_idx_list)

d.train_model(m, x1, x2, y1, y2, vx1, vx2, vy1, vy2, ex1, ex2, ey1, ey2)
