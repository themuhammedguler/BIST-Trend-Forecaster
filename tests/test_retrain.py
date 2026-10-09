# tests/test_retrain.py - Model yeniden eğitimi ve kalite kapısı testleri (#20)
import os

import pytest

import config
import model_train

pytestmark = pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="veri seti yok")


def test_train_model_aborts_when_accuracy_below_threshold(small_dataset):
    """Model doğruluğu belirlenen kalite eşiğinin altındaysa model
    kaydedilmeden hata fırlatılmalıdır."""
    with pytest.raises(ValueError, match="minimum kalite eşiğinin"):
        model_train.train_model(min_accuracy=0.999)

    # Model ve metaveri dosyası yazılmamalı
    assert not os.path.exists(config.MODEL_PATH)
    assert not os.path.exists(config.MODEL_META_PATH)


def test_train_model_succeeds_when_accuracy_meets_threshold(small_dataset):
    """Model doğruluğu kalite eşiğini aştığında başarıyla kaydedilmelidir."""
    model, acc = model_train.train_model(min_accuracy=0.40)
    assert acc >= 0.40
    assert os.path.exists(config.MODEL_PATH)
    assert os.path.exists(config.MODEL_META_PATH)
