from pathlib import Path
import io, zipfile, pyzipper, pandas as pd
from etl_modules.io_utils import read_csv_sniff

def test_read_csv_zipcrypto(tmp_path: Path):
    # cria zip com ZipCrypto
    csv_bytes = b"col1,col2\n1,2\n"
    zpath = tmp_path / "z.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Emotion_2025.csv", csv_bytes)
    df = read_csv_sniff(zpath, "Emotion*.csv")
    assert df.shape == (1,2)

def test_read_csv_aes(tmp_path: Path):
    csv_bytes = b"emotion,score\nPositive,1\n"
    zpath = tmp_path / "zaes.zip"
    with pyzipper.AESZipFile(zpath, "w", encryption=pyzipper.WZ_AES) as zf:
        zf.setpassword(b"secret")
        zf.writestr("Emotion_2025.csv", csv_bytes)
    df = read_csv_sniff(zpath, "Emotion*.csv", password="secret")
    assert list(df.columns) == ["emotion","score"]
