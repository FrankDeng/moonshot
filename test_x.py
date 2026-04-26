import tweepy

# # 填入你在 Developer Portal 获取的四个凭据
# API_KEY = "你的_API_KEY"
# API_SECRET = "你的_API_SECRET"
# ACCESS_TOKEN = "你的_ACCESS_TOKEN"
# ACCESS_TOKEN_SECRET = "你的_ACCESS_TOKEN_SECRET"
API_KEY= "REV1VE5MV3lyS3JqUU1FSGFzTzY6MTpjaQ"
API_SECRET="zztCs2PtjAyAd76WvX2oMRn3X23BLNdq60ukEPATA6UmSRlSDC"
ACCESS_TOKEN="2048413890489589761-ZXDziVvOqCYeHd2a2s38Y1sFwZyGwg"
ACCESS_TOKEN_SECRET="gNePPwIkpXj08g1MfGp1N2qGYxgcHJ5eq2EIZ0OnNXtLR"


def post_tweet(text):

    client = tweepy.Client(
        consumer_key=API_KEY,
        consumer_secret=API_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )
        
        # 2. 发布推文
    response = client.create_tweet(text=text)
    print(f"✅ 发布成功！推文 ID: {response.data['id']}")

if __name__ == "__main__":
    # 建议带上时间戳，避免因发送重复内容被 X 判定为垃圾信息
    import datetime
    msg = f"Test post from API v2 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    post_tweet(msg)