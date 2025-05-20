import pandas as pd
import os

def convert_parquet_to_csv(parquet_file_path: str, csv_file_path: str):
    """
    지정된 Parquet 파일을 CSV 파일로 변환하여 저장합니다.

    Args:
        parquet_file_path (str): 입력 Parquet 파일의 전체 경로.
        csv_file_path (str): 출력 CSV 파일을 저장할 전체 경로.
    """
    if not os.path.exists(parquet_file_path):
        print(f"Error: 입력 Parquet 파일 '{parquet_file_path}'을(를) 찾을 수 없습니다.")
        return

    try:
        print(f"'{parquet_file_path}' 파일을 읽는 중...")
        df = pd.read_parquet(parquet_file_path, engine='pyarrow')
        print("Parquet 파일 읽기 완료.")

        # CSV 파일 저장 경로의 디렉토리 확인 및 생성
        output_dir = os.path.dirname(csv_file_path)
        if output_dir and not os.path.exists(output_dir): # output_dir이 빈 문자열이 아닌 경우 (상대경로로 현재 디렉토리에 저장시)
            os.makedirs(output_dir)
            print(f"디렉토리 '{output_dir}'를 생성했습니다.")

        print(f"'{csv_file_path}' 파일로 저장하는 중...")
        # CSV 저장 시 한글 깨짐 방지 및 Excel 호환성을 위해 'utf-8-sig' 인코딩 사용 권장
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"데이터가 성공적으로 '{csv_file_path}'에 CSV 파일로 저장되었습니다.")

    except ImportError:
        print("Error: 'pyarrow' 라이브러리가 필요합니다. 'pip install pyarrow'로 설치해주세요.")
    except Exception as e:
        print(f"Error: 파일 변환 중 오류 발생 - {e}")

if __name__ == '__main__':
    # --- 경로 설정 ---
    # Parquet 파일이 있는 기본 경로 (이전 스크립트의 data_folder_path 와 유사하게 설정)
    # 예: data_folder_path = './data/'
    # 여기서는 Parquet 파일이 'data/processed/' 디렉토리에 있다고 가정합니다.
    
    base_data_path = 'data' # 현재 스크립트 위치 기준 'data' 폴더
    processed_folder = os.path.join(base_data_path, 'processed')

    input_parquet_file = os.path.join(processed_folder, 'clean_master.parquet')
    output_csv_file = os.path.join(processed_folder, 'clean_master.csv')

    # --- 변환 실행 ---
    convert_parquet_to_csv(parquet_file_path=input_parquet_file, csv_file_path=output_csv_file)