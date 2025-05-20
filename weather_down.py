import requests
import pandas as pd
import time
from datetime import datetime as dt # dt 별칭으로 임포트

# 보스턴 마라톤 정보
BOSTON_LATITUDE = 42.3601
BOSTON_LONGITUDE = -71.0589

MARATHON_DATES = {
    2015: "2015-04-20",
    2016: "2016-04-18",
    2017: "2017-04-17",
}

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# 레이스 시간 창 (현지 시간 기준, 24시간 형식)
RACE_START_HOUR = 9  # 오전 9시
RACE_END_HOUR = 16 # 오후 4시 (16시까지 포함)

def fetch_race_window_weather(year, date_str):
    """
    지정된 날짜의 특정 시간대(레이스 시간) 평균 기온 및 습도를 가져옵니다.
    시간별 데이터를 요청하여 해당 시간대의 평균을 계산합니다.
    """
    params = {
        "latitude": BOSTON_LATITUDE,
        "longitude": BOSTON_LONGITUDE,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ["temperature_2m", "relativehumidity_2m"], # 시간별 기온과 습도 요청
        "timezone": "America/New_York"
    }

    try:
        response = requests.get(API_URL, params=params)
        print(f"Fetching URL for hourly weather ({year}): {response.url}")
        response.raise_for_status()
        data = response.json()

        hourly_data = data.get('hourly', {})
        time_stamps_iso = hourly_data.get('time', [])
        temperatures_hourly = hourly_data.get('temperature_2m', [])
        humidities_hourly = hourly_data.get('relativehumidity_2m', [])

        if not (time_stamps_iso and temperatures_hourly and humidities_hourly):
            print(f"Warning: {year}년 시간별 데이터를 가져오는 데 실패했거나 일부 데이터가 비어있습니다.")
            if 'reason' in data: print(f"API Error Reason: {data['reason']}")
            return None, None # 기온, 습도 모두 None 반환

        race_window_temps = []
        race_window_humidities = []

        for i, time_iso in enumerate(time_stamps_iso):
            try:
                # ISO 8601 형식의 문자열에서 시간(hour) 정보 추출
                current_hour = dt.fromisoformat(time_iso).hour
                
                if RACE_START_HOUR <= current_hour <= RACE_END_HOUR:
                    if temperatures_hourly[i] is not None:
                        race_window_temps.append(temperatures_hourly[i])
                    if humidities_hourly[i] is not None:
                        race_window_humidities.append(humidities_hourly[i])
            except IndexError:
                print(f"Warning: {year}년 데이터 처리 중 인덱스 오류 발생. (시간: {len(time_stamps_iso)}, 기온: {len(temperatures_hourly)}, 습도: {len(humidities_hourly)})")
                continue # 다음 시간 데이터 처리
            except Exception as e_parse:
                print(f"Warning: {year}년 시간 파싱 또는 데이터 접근 중 오류: {e_parse} (time_iso: {time_iso})")
                continue


        avg_temp = sum(race_window_temps) / len(race_window_temps) if race_window_temps else None
        avg_humidity = sum(race_window_humidities) / len(race_window_humidities) if race_window_humidities else None
        
        if avg_temp is None or avg_humidity is None:
            print(f"Warning: {year}년 레이스 시간 창 ({RACE_START_HOUR:02d}:00-{RACE_END_HOUR:02d}:00) 내 유효한 기온 또는 습도 데이터 부족.")

        return avg_temp, avg_humidity

    except requests.exceptions.HTTPError as http_err:
        print(f"Error: {year}년 시간별 날씨 데이터 HTTP 오류: {http_err.response.status_code}")
        if response is not None and response.text: print(f"Response content: {response.text}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred for {year} (hourly weather fetch): {e}")
        return None, None

def create_weather_csv(output_filename="boston_marathon_weather_race_hours.csv"): # 파일명 변경
    all_weather_data = []
    for year, date_str in MARATHON_DATES.items():
        print(f"\n{year}년 ({date_str}) 레이스 시간대 날씨 데이터 수집 중...")
        
        avg_temp, avg_humidity = fetch_race_window_weather(year, date_str)
        time.sleep(0.5) # API 서버 부하 감소를 위한 지연
        
        if avg_temp is not None and avg_humidity is not None:
            all_weather_data.append({
                "year": year,
                "date": date_str,
                "temperature_race": avg_temp,
                "humidity_race": avg_humidity
            })
            print(f"Data for {year} (Race Window Avg {RACE_START_HOUR:02d}-{RACE_END_HOUR:02d}): Temp={avg_temp:.2f}°C, Humidity={avg_humidity:.2f}%")
        else:
            temp_val_str = f"{avg_temp:.2f}°C" if avg_temp is not None else "None"
            humidity_val_str = f"{avg_humidity:.2f}%" if avg_humidity is not None else "None"
            print(f"Warning: {year}년의 레이스 시간대 평균 기온 또는 습도 데이터를 완전하게 가져오지 못했습니다 (Temp: {temp_val_str}, Humidity: {humidity_val_str}).")

    if not all_weather_data:
        print("수집된 날씨 데이터가 없습니다. CSV 파일을 생성하지 않습니다.")
        return

    weather_df = pd.DataFrame(all_weather_data)
    weather_df['year'] = weather_df['year'].astype(int)

    try:
        weather_df.to_csv(output_filename, index=False)
        print(f"\n레이스 시간대 평균 날씨 데이터가 성공적으로 '{output_filename}' 파일로 저장되었습니다.")
        if not weather_df.empty:
            print("생성된 파일 내용 미리보기:")
            print(weather_df.head())
    except IOError:
        print(f"Error: '{output_filename}' 파일 저장 중 오류가 발생했습니다.")

if __name__ == "__main__":
    # 생성될 CSV 파일명 (레이스 시간대 평균임을 명시)
    weather_output_file = "boston_marathon_weather_race_hours.csv" 
    create_weather_csv(output_filename=weather_output_file)

    print(f"\n이제 이 생성된 '{weather_output_file}' 파일을 사용하여 'merge_weather_data' 함수에 전달하세요.")