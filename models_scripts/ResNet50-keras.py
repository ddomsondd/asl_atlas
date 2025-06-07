import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, 
    Dense, 
    BatchNormalization, 
    Activation, 
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# # === 1. Bazowy model (ResNet50 bez głowicy) ===
# # Zamrażamy całą sieć, bo uczymy tylko nową głowicę
# base = ResNet50(
#     include_top=False,        # usuwamy oryginalną "główkę"
#     weights="imagenet",       # pretrenowane wagi
#     input_shape=(224,224,3)
# )
# for layer in base.layers:
#     layer.trainable = False   #  wszystkie warstwy bazowe zamrożone

# # === 2. Budowa nowej głowicy ===
# x = base.output

# # — Globalna agregacja cech
# x = GlobalAveragePooling2D()(x)

# # — 1) BatchNorm → 2) Activation → 3) Dropout
# # BatchNorm normalizuje aktywacje, co stabilizuje uczenie i przyspiesza konwergencję  
# #    (Ioffe & Szegedy, 2015).
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# # Dropout losowo wyłącza neurony, by zapobiec overfittingowi  
# #    (Srivastava et al., 2014).
# x = Dropout(0.5)(x)

# # — stopniowe zmniejszanie liczby neuronów w warstwach Dense
# #    256 → 128 → 64 → 32, by unikać nadmiernej liczby parametrów
# #    i wymusić uczenie coraz bardziej abstrakcyjnych reprezentacji.

# # 2nd Dense block
# x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(0.4)(x)

# # 3rd Dense block
# x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(0.3)(x)

# # 4th Dense block
# x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(0.2)(x)

# # 5th Dense block
# x = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(0.1)(x)

# # — warstwa wyjściowa
# outputs = Dense(29, activation='softmax')(x)  # 29 klas ASL

# # Połączenie bazy i głowicy
# model = Model(inputs=base.input, outputs=outputs)

# # === 3. Kompilacja ===
# Używamy Adam + umiarkowany lr (1e-4) bo uczymy tylko głowicę
# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # === 4. Przygotowanie danych ===
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     width_shift_range=0.1, height_shift_range=0.1,
#     brightness_range=[0.8,1.2],
#     validation_split=0.2
# )
# train_gen = train_datagen.flow_from_directory(
#     "asl_alphabet/asl_alphabet_train/asl_alphabet_train",
#     target_size=(224,224), batch_size=32,
#     class_mode="categorical", subset="training"
# )
# val_gen = train_datagen.flow_from_directory(
#     "asl_alphabet/asl_alphabet_train/asl_alphabet_train",
#     target_size=(224,224), batch_size=32,
#     class_mode="categorical", subset="validation"
# )

# # === 5. Callbacki ===
# callbacks = [
#     ModelCheckpoint("model_feature_extraction.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
#     EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
# ]

# # === 6. Trenowanie głowicy ===
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=10,
#     callbacks=callbacks
# )

model = load_model("/kaggle/input/57epoch/keras/default/1/model_after_finetuned-day5-epoch57.keras")

for layer in model.layers:
    layer.trainable = False

resnet_layers = [layer for layer in model.layers if layer.name.startswith(("conv", "bn"))]
for layer in resnet_layers[-30:]:
    layer.trainable = True


set_trainable = False
for layer in model.layers:
    if layer.name.startswith("conv5_block1"):  # punkt startowy odmrażania ResNet
        set_trainable = True
    if set_trainable:
        layer.trainable = True

# Kompilacja z małym learning rate
model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.3],
    channel_shift_range=10.0,  
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

callbacks = [
    ModelCheckpoint("/kaggle/working/model_after_finetuned-day5.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=65,
    initial_epoch=57,
    callbacks=callbacks
)