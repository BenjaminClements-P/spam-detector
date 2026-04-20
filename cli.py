"""
CLI tool — test emails directly from command line
Usage: python cli.py
       python cli.py "Your email text here"
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.spam_detector import SpamDetector
from train import SPAM_SAMPLES, HAM_SAMPLES

def ensure_trained(detector):
    if not detector.is_trained:
        print("[*] Training model (first run)...")
        texts  = SPAM_SAMPLES + HAM_SAMPLES
        labels = ["spam"] * len(SPAM_SAMPLES) + ["ham"] * len(HAM_SAMPLES)
        detector.train(texts, labels)

def print_result(result, email_text):
    label = result['label'].upper()
    conf  = result['confidence'] * 100
    spam_pct = result.get('spam_prob', 0) * 100
    ham_pct  = result.get('ham_prob',  0) * 100

    bar_len = 30
    spam_bar = '█' * int(spam_pct / 100 * bar_len) + '░' * (bar_len - int(spam_pct / 100 * bar_len))
    ham_bar  = '█' * int(ham_pct  / 100 * bar_len) + '░' * (bar_len - int(ham_pct  / 100 * bar_len))

    icon = "🚨" if label == "SPAM" else "✅"
    print(f"\n{'─'*55}")
    print(f"  {icon}  VERDICT: {label}  ({conf:.1f}% confident)")
    print(f"{'─'*55}")
    print(f"  SPAM [{spam_bar}] {spam_pct:.1f}%")
    print(f"  HAM  [{ham_bar}] {ham_pct:.1f}%")
    if result.get('features'):
        top_words = [f['word'] for f in result['features'][:6]]
        print(f"\n  Key NLP features: {', '.join(top_words)}")
    if result.get('cached'):
        print("  ℹ️  Result from hash table cache")
    print(f"{'─'*55}\n")

def main():
    detector = SpamDetector()
    ensure_trained(detector)

    if len(sys.argv) > 1:
        email_text = ' '.join(sys.argv[1:])
        result = detector.predict(email_text)
        print_result(result, email_text)
        return

    print("\n🛡️  SpamShield AI — Command Line Interface")
    print("   Data Structure: Hash Table | AI: Naive Bayes + NLP")
    print("   Type 'quit' to exit | Type 'stats' for index stats\n")

    while True:
        try:
            text = input("📧 Enter email text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text.lower() == 'quit':
            break
        if text.lower() == 'stats':
            s = detector.email_index.stats()
            print(f"   Hash table: {s['total']} emails indexed | {s['spam']} spam | {s['ham']} legitimate | {s['feedback_count']} feedback entries\n")
            continue
        if not text:
            continue
        result = detector.predict(text)
        print_result(result, text)

        # Feedback
        fb = input("   Correct? [s=spam / h=ham / Enter to skip]: ").strip().lower()
        if fb in ('s', 'spam'):
            detector.learn_from_feedback(text, 'spam')
            print("   ✓ Model updated — marked as SPAM\n")
        elif fb in ('h', 'ham'):
            detector.learn_from_feedback(text, 'ham')
            print("   ✓ Model updated — marked as LEGITIMATE\n")

if __name__ == '__main__':
    main()
