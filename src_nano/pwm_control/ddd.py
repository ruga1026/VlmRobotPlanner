import time
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

# 1) I2C 및 PCA9685 초기화
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # 50Hz → period = 20_000 μs

neutral = 1800

# 2) μs 단위 펄스를 duty_cycle(16bit)로 변환해 설정하는 헬퍼 함수
PERIOD_US = 1_000_000 // pca.frequency  # = 20_000
def set_us(ch: int, us: int):
    """채널 ch에 us(μs) 만큼 PWM 펄스를 보낸다."""
    # us 범위 체크 (안전장치)
    if not 1000 <= us - (neutral - 1500) <= 2000:
        print(f"⚠️ {us}μs는 허용 범위(1000–2000μs)를 벗어납니다.")
        return
    duty = int(us * 0xFFFF / PERIOD_US)
    pca.channels[ch].duty_cycle = duty

# 3) 기본 중립 신호(1500μs)로 초기화
set_us(15, neutral)  # 일반 서보 (16번째 채널)
set_us(14, neutral)  # ESC       (15번째 채널)
time.sleep(0.5)

print("Enter에 따라 PWM(μs)을 설정합니다. 종료는 ‘q’ 또는 ‘quit’ 입력.")

try:
    while True:
        val = input("▶ PWM 입력 (1000–2000) 또는 q: ").strip()
        if val.lower() in ('q', 'quit'):
            break
        # 숫자로 변환 시도
        try:
            us = int(val) + (neutral - 1500)
        except ValueError:
            print("⚠️ 숫자를 입력하거나, ‘q’로 종료하세요.")
            continue

        # 양쪽 장치에 같은 PWM 신호 전송
        set_us(15, us)  # 서보
        set_us(14, us)  # ESC
        print(f"✅ 채널15(서보), 채널14(ESC)에 {us}μs 신호 설정")

    # 루프 빠져나오면 중립으로 복귀
    print("중립(1500μs)으로 복귀합니다.")
    set_us(15, neutral)
    set_us(14, neutral)
    time.sleep(0.2)

finally:
    pca.deinit()
    print("PCA9685 비활성화 완료.")