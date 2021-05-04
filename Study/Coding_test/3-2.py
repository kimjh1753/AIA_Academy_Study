# 3-2.py

# N, M , K를 공백으로 구분하여 입력받기
n, m, k = map(int, input().split()) # n : 배열의 크기, m : 숫자가 더해지는 횟수, k : 동빈이의 큰 수의 법칙

# N개의 수를 공백으로 구분하여 입력받기
data = list(map(int, input().split()))

data.sort() # 입력받은 수 정렬
first = data[n - 1] # 가장 큰 수
second = data[n - 2] # 두 번째로 큰 수

# 가장 큰 수가 더해지는 횟수 계산
count = int(m / (k + 1)) * k
count += m % (k + 1)

result = 0
result += (count) * first # 가장 큰 수 더하기
result += (m - count) * second # 두 번째로 큰 수 더하기

print(result)