
# dataset = TensorDataset(x, x_face, y_label_signal, y_label_class)
# del x, x_face, y_label_signal, y_label_class


# # train, validation and test
# train_size = int(0.8 * len(dataset))
# val_size = int(0.2 * len(dataset))

# train_set, val_set = random_split(dataset, [train_size, val_size])

# # train, validation and test dataloader
# train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

# del train_set

# val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

# del val_set

# # model
# model = Net().cuda()

# # # # callbacks
# checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss',
#     dirpath='/content/drive/MyDrive/Neuro/facial_recognition_preprocessed_data/architecture/bin',
#     filename='face_recognition-tfr-{epoch:02d}-{val_loss:.2f}',
#     save_top_k=1,
#     mode='min',
# )

# early_stop_callback = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.00,
#     patience=12,
#     verbose=True,
#     mode='min'
# )

# # trainer
# trainer = pl.Trainer(
#     accelerator='auto',
#     callbacks=[checkpoint_callback, early_stop_callback],
#     max_epochs=50
# )

# # train
# trainer.fit(model, train_loader, val_loader)