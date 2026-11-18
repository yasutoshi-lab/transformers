"""freesia-supervised-cl の classification で使う話題ラベル（ja と en は同じ並びで対応）。"""
LABELS = [("政治", "Politics"), ("経済・ビジネス", "Economy and business"), ("法律・司法", "Law and justice"),
          ("社会・事件", "Society and incidents"), ("国際関係", "International relations"), ("歴史", "History"),
          ("地理・地域", "Geography and places"), ("物理・化学", "Physics and chemistry"),
          ("生物・生命科学", "Biology and life sciences"), ("医療・健康", "Medicine and health"),
          ("情報技術", "Information technology"), ("工学・製造", "Engineering and manufacturing"),
          ("環境・自然", "Environment and nature"), ("宇宙・天文", "Space and astronomy"), ("数学", "Mathematics"),
          ("教育", "Education"), ("宗教・哲学", "Religion and philosophy"), ("文学", "Literature"), ("言語", "Language"),
          ("音楽", "Music"), ("映画・テレビ", "Film and television"), ("美術・建築", "Art and architecture"),
          ("ゲーム・アニメ・漫画", "Games, anime and comics"), ("スポーツ", "Sports"), ("食・料理", "Food and cooking"),
          ("旅行・観光", "Travel and tourism"), ("暮らし・家庭", "Home and lifestyle"), ("交通", "Transportation"),
          ("軍事", "Military"), ("人物", "Biography")]
LABEL_LIST = {"ja": [a for a, _ in LABELS], "en": [b for _, b in LABELS]}
