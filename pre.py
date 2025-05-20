import pandas as pd
import numpy as np
import os # 파일 경로 및 디렉토리 생성을 위해 추가

def load_marathon_data(data_path: str = './data/') -> pd.DataFrame:
    """
    보스턴 마라톤 2015, 2016, 2017년도 CSV 데이터를 로드하고 병합합니다.

    Args:
        data_path (str): CSV 파일들이 위치한 기본 경로입니다.

    Returns:
        pd.DataFrame: 3개년치 데이터가 병합된 데이터프레임.
                       'Year' 컬럼이 추가되어 있습니다.
    """
    try:
        df_15 = pd.read_csv(f'{data_path}marathon_results_2015.csv')
        df_16 = pd.read_csv(f'{data_path}marathon_results_2016.csv')
        df_17 = pd.read_csv(f'{data_path}marathon_results_2017.csv')
    except FileNotFoundError as e:
        print(f"Error: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요: {e.filename}")
        raise

    df_15['Year'] = 2015
    df_16['Year'] = 2016
    df_17['Year'] = 2017

    df_combined = pd.concat([df_15, df_16, df_17], ignore_index=True)
    
    # 'Unnamed: 0' 컬럼이 존재하면 제거 (CSV 저장 시 생성된 불필요한 인덱스일 수 있음)
    if 'Unnamed: 0' in df_combined.columns:
        df_combined = df_combined.drop(columns=['Unnamed: 0'])
        
    return df_combined

def process_schema_and_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 스키마를 매핑하고 시간 관련 컬럼들을 초 단위로 변환합니다.

    Args:
        df (pd.DataFrame): 원본 마라톤 데이터프레임.

    Returns:
        pd.DataFrame: 스키마 매핑 및 시간 변환이 완료된 데이터프레임.
    """
    processed_df = df.copy()

    # 변환할 시간 컬럼 목록 (원본 데이터셋의 실제 컬럼명 기준)
    time_columns_original = ['5K', '10K', '15K', '20K', 'Half', '25K', '30K', '35K', '40K', 'Official Time']
    
    # 새로운 컬럼명 (초 단위 변환 후)
    time_columns_seconds = [
        'split_5k_sec', 'split_10k_sec', 'split_15k_sec', 'split_20k_sec', 
        'split_half_sec', 'split_25k_sec', 'split_30k_sec', 'split_35k_sec', 
        'split_40k_sec', 'finish_time_sec'
    ]

    for original_col, new_col_sec in zip(time_columns_original, time_columns_seconds):
        if original_col in processed_df.columns:
            # '-' 값을 NaN으로 변환
            processed_df[original_col] = processed_df[original_col].replace('-', np.nan)
            # 시간 문자열을 초 단위로 변환 (H:M:S, M:S 등의 형식 지원)
            # 변환 불가능한 값은 NaT가 되고, .dt.total_seconds() 호출 시 NaN이 됨
            processed_df[new_col_sec] = pd.to_timedelta(processed_df[original_col], errors='coerce').dt.total_seconds()
        else:
            print(f"Warning: 원본 시간 컬럼 '{original_col}'을(를) 찾을 수 없습니다. 해당 컬럼 처리를 건너<0xEB><0x84><0xB5>니다.")


    # 필요한 기본 정보 컬럼 선택 및 이름 변경
    # 'Bib'는 숫자로 된 ID일 수 있으므로 형변환 고려 (여기서는 그대로 둠)
    columns_to_keep_map = {
        'Bib': 'bib',
        'Age': 'age',      # 'Age'는 보통 정수형이므로 그대로 사용 가능
        'M/F': 'sex',      # 'M/F'를 'sex'로 변경
        'Year': 'year'     # 'Year'는 이미 존재하며 정수형
    }
    
    final_columns = []
    for original_name, new_name in columns_to_keep_map.items():
        if original_name in processed_df.columns:
            processed_df.rename(columns={original_name: new_name}, inplace=True)
            final_columns.append(new_name)
        else:
            print(f"Warning: 기본 정보 컬럼 '{original_name}'을(를) 찾을 수 없습니다.")

    # 변환된 시간 컬럼들 추가
    for col_sec in time_columns_seconds:
        if col_sec in processed_df.columns:
            final_columns.append(col_sec)
    
    # 정의된 최종 컬럼들만 선택하여 반환
    processed_df = processed_df[final_columns]

    # 데이터 타입 정리 (선택 사항이나 권장)
    if 'age' in processed_df.columns:
        processed_df['age'] = pd.to_numeric(processed_df['age'], errors='coerce') # coerce로 변환 실패 시 NaN 처리
    # 'bib'도 숫자형이라면 pd.to_numeric 적용 가능

    return processed_df
def merge_weather_data(marathon_df: pd.DataFrame, weather_file_path: str) -> pd.DataFrame:
    """
    마라톤 데이터에 날씨 데이터를 'year' 기준으로 병합합니다.

    Args:
        marathon_df (pd.DataFrame): 스키마 및 시간 처리가 완료된 마라톤 데이터프레임.
        weather_file_path (str): 날씨 데이터 CSV 파일 경로.

    Returns:
        pd.DataFrame: 날씨 데이터가 병합된 마라톤 데이터프레임.
    """
    try:
        weather_df = pd.read_csv(weather_file_path)
    except FileNotFoundError:
        print(f"Error: 날씨 데이터 파일({weather_file_path})을 찾을 수 없습니다. 경로를 확인해주세요.")
        # 날씨 파일이 없어도 처리를 계속하고 싶다면 빈 데이터프레임 또는 None을 반환하도록 수정 가능
        raise

    # 날씨 데이터의 'year' 컬럼 확인 및 데이터 타입 통일 (marathon_df의 'year'는 정수형 가정)
    # temp.csv의 연도 컬럼명이 'year'가 아니라면 아래에서 실제 컬럼명으로 변경해야 합니다.
    # 예: 만약 'date' 컬럼에서 연도를 추출해야 한다면:
    # weather_df['year'] = pd.to_datetime(weather_df['date']).dt.year
    if 'year' not in weather_df.columns:
        # 'time' 컬럼에서 연도 추출 시도 (Open-Meteo 기본 출력 형식 가정)
        if 'time' in weather_df.columns:
            weather_df['year'] = pd.to_datetime(weather_df['time']).dt.year
        else:
            print("Error: 날씨 데이터에 'year' 또는 'time' 컬럼이 없습니다. 연도 정보를 추출할 수 없습니다.")
            return marathon_df # 또는 예외 발생

    # 날씨 데이터의 기온 및 습도 컬럼명을 표준화 (예시 컬럼명, 실제 파일에 맞게 수정)
    # Open-Meteo 기본 컬럼명 예시: 'temperature_2m_mean', 'relativehumidity_2m_mean'
    weather_rename_map = {
        'temperature_2m_mean': 'temperature_race', # 실제 temp.csv의 기온 컬럼명
        'relativehumidity_2m_mean': 'humidity_race'  # 실제 temp.csv의 습도 컬럼명
    }
    
    # 실제 존재하는 컬럼만 리네임 시도
    cols_to_rename_existing = {k: v for k, v in weather_rename_map.items() if k in weather_df.columns}
    if not cols_to_rename_existing:
         print(f"Warning: 날씨 데이터({weather_file_path})에서 표준 기온/습도 컬럼을 찾을 수 없습니다.")
    weather_df.rename(columns=cols_to_rename_existing, inplace=True)

    # 필요한 컬럼만 선택 (year와 표준화된 기온/습도 컬럼)
    required_weather_cols = ['year']
    if 'temperature_race' in weather_df.columns:
        required_weather_cols.append('temperature_race')
    if 'humidity_race' in weather_df.columns:
        required_weather_cols.append('humidity_race')
    
    if len(required_weather_cols) <= 1 : # year 외에 유효한 날씨 컬럼이 없는 경우
        print(f"Warning: 날씨 데이터({weather_file_path})에 유효한 기온 또는 습도 데이터가 부족하여 병합하지 않습니다.")
        return marathon_df

    weather_df_selected = weather_df[required_weather_cols].drop_duplicates(subset=['year'], keep='first')

    # 마라톤 데이터와 날씨 데이터 병합
    # marathon_df의 'year' 컬럼은 process_schema_and_times 함수에서 이미 생성 및 정수형으로 처리됨
    merged_df = pd.merge(marathon_df, weather_df_selected, on='year', how='left')
    
    print(f"날씨 데이터 병합 완료. 추가된 컬럼: {', '.join(weather_df_selected.columns.drop('year')) if len(weather_df_selected.columns) > 1 else '없음'}")
    return merged_df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    마라톤 데이터에 특성 엔지니어링을 적용합니다.
    - 구간별 페이스 계산 (초/km)
    - 기온 보정치 계산 (초/km)
    - 사용자 입력용 컬럼 추가

    Args:
        df (pd.DataFrame): 날씨 데이터가 병합된 마라톤 데이터프레임.

    Returns:
        pd.DataFrame: 특성 엔지니어링이 완료된 데이터프레임.
    """
    engineered_df = df.copy()

    # 1. 구간별 페이스 계산 (초/km)
    # 각 구간은 5km로 가정합니다.
    # split_XXk_sec 컬럼들이 이미 초 단위로 변환되어 있다고 가정합니다.
    
    split_cols_in_seconds = {
        0: None, # 시작점
        5: 'split_5k_sec',
        10: 'split_10k_sec',
        15: 'split_15k_sec',
        20: 'split_20k_sec',
        25: 'split_25k_sec',
        30: 'split_30k_sec',
        35: 'split_35k_sec',
        40: 'split_40k_sec',
    }
    
    segment_distances_km = [5, 10, 15, 20, 25, 30, 35, 40] # 각 스플릿 지점까지의 누적 거리

    last_split_time_col = None # 이전 스플릿의 시간 (0초에서 시작)
    last_split_distance_km = 0

    for current_distance_km in segment_distances_km:
        current_split_time_col = split_cols_in_seconds.get(current_distance_km)
        
        if current_split_time_col and current_split_time_col in engineered_df.columns:
            segment_label = f"{last_split_distance_km}_{current_distance_km}km"
            pace_col_name = f"pace_{segment_label}_per_km"

            if last_split_time_col: # 0-5km 이후 구간
                # 현재 구간에 걸린 시간 = 현재 스플릿 시간 - 이전 스플릿 시간
                segment_duration_sec = engineered_df[current_split_time_col] - engineered_df[last_split_time_col]
            else: # 0-5km 구간
                segment_duration_sec = engineered_df[current_split_time_col]
            
            segment_length_km = current_distance_km - last_split_distance_km
            
            if segment_length_km > 0:
                engineered_df[pace_col_name] = segment_duration_sec / segment_length_km
            else:
                engineered_df[pace_col_name] = np.nan # 거리가 0이거나 음수일 경우 방지

            last_split_time_col = current_split_time_col # 다음 계산을 위해 현재 스플릿 정보 저장
        
        last_split_distance_km = current_distance_km
    
    print("구간별 페이스 계산 완료.")

    # 2. 기온 보정치 계산 (초/km)
    # 최적 기온 T_opt = 12 °C
    # Δpace (sec/km) = 0.8 * (T_race – 12)  (T_race > 12 °C 일 때)
    if 'temperature_race' in engineered_df.columns:
        T_opt = 12.0
        penalty_factor = 0.8
        
        # temperature_race가 NaN이 아닌 경우에만 계산
        temp_penalty = np.where(
            (engineered_df['temperature_race'].notna()) & (engineered_df['temperature_race'] > T_opt),
            penalty_factor * (engineered_df['temperature_race'] - T_opt),
            0.0 # 12도 이하이거나 기온 정보가 없으면 페널티 없음
        )
        engineered_df['temp_penalty_sec_per_km'] = temp_penalty
        print("기온 보정치 계산 완료.")
    else:
        print("Warning: 'temperature_race' 컬럼이 없어 기온 보정치를 계산할 수 없습니다.")
        engineered_df['temp_penalty_sec_per_km'] = 0.0 # 컬럼은 만들어두되 값은 0으로

    # 3. 사용자 입력용 컬럼 확보 (NaN으로 초기화)
    engineered_df['user_weight_kg'] = np.nan
    engineered_df['user_weekly_km'] = np.nan
    engineered_df['user_target_time_sec'] = np.nan
    print("사용자 입력용 컬럼 추가 완료.")

    return engineered_df

def save_to_parquet(df: pd.DataFrame, output_path: str):
    """
    데이터프레임을 Parquet 파일로 저장합니다.
    저장 경로의 디렉토리가 없으면 생성합니다.

    Args:
        df (pd.DataFrame): 저장할 데이터프레임.
        output_path (str): Parquet 파일을 저장할 전체 경로.
    """
    try:
        # 저장 경로의 디렉토리 확인 및 생성
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"디렉토리 '{output_dir}'를 생성했습니다.")

        df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"데이터가 성공적으로 '{output_path}'에 Parquet 파일로 저장되었습니다.")
    except ImportError:
        print("Error: 'pyarrow' 라이브러리가 필요합니다. 'pip install pyarrow'로 설치해주세요.")
    except Exception as e:
        print(f"Error: Parquet 파일 저장 중 오류 발생 - {e}")

if __name__ == '__main__':
    # 데이터 파일이 저장된 경로 (사용자 환경에 맞게 수정)
    data_folder_path = './data/' 
    weather_data_file = 'boston_marathon_weather.csv' # 날씨 데이터 파일명
    
    # ... (1단계, 2단계, 3단계 호출 코드 생략 - 이전과 동일) ...
    print(f"{data_folder_path} 에서 마라톤 데이터를 로드합니다...")
    raw_df = load_marathon_data(data_path=data_folder_path)
    print("마라톤 데이터 로드 완료.")

    print("\n1단계: 스키마 매핑 및 시간 변환을 시작합니다...")
    processed_df_step1 = process_schema_and_times(raw_df)
    print("1단계 처리 완료.")

    print("\n2단계: 날씨 데이터 병합을 시작합니다...")
    merged_df_step2 = merge_weather_data(processed_df_step1, weather_file_path=weather_data_file)
    print("2단계 처리 완료.")
    
    print("\n3단계: 특성 엔지니어링을 시작합니다...")
    final_df_step3 = feature_engineering(merged_df_step2)
    print("3단계 처리 완료.")
    # (이하 기존 print문들 생략)
    # print("\n처리된 데이터 정보 (3단계):")
    # final_df_step3.info()
    # print("\n처리된 데이터 첫 5행 (3단계):")
    # print(final_df_step3.head())


    # --- 여기부터 4단계 코드 추가 ---
    print("\n4단계: 최종 Parquet 파일 저장을 시작합니다...")
    # 프로젝트 요약의 저장소 구조에 따름: data/processed/clean_master.parquet
    # data_folder_path가 './data/' 이므로, os.path.join을 사용하거나 직접 경로 조합
    processed_data_dir = os.path.join(data_folder_path, 'processed') 
    output_parquet_path = os.path.join(processed_data_dir, 'clean_master.parquet')
    
    save_to_parquet(final_df_step3, output_path=output_parquet_path)
    # --- 여기까지 4단계 코드 추가 ---

    print("\n모든 전처리 파이프라인이 완료되었습니다.")