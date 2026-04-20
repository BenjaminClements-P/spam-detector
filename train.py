"""
Train the Spam Detection model with a built-in dataset.
Run: python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.spam_detector import SpamDetector
from sklearn.model_selection import train_test_split

# ─── Built-in labeled dataset ────────────────────────────────────────────────
SPAM_SAMPLES = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
    "FREE entry in 2 a weekly competition to win FA Cup final tkts. Text FA to 87121",
    "URGENT: Your account has been compromised. Click this link immediately to secure it.",
    "You have been selected for a special prize. Call now to claim your free vacation.",
    "Get rich quick! Make $5000 per week working from home. Limited offer!",
    "Win a brand new iPhone 15! You are our lucky winner today. Claim your prize now!",
    "CHEAP Viagra online! Buy now and get 90% discount. No prescription needed.",
    "Earn money fast! No experience required. Work from home guaranteed income.",
    "Your PayPal account is suspended. Verify your identity now to avoid permanent closure.",
    "Hot singles in your area want to meet you. Click here for free access.",
    "Congratulations winner! Claim your cash prize of $10,000 before it expires!",
    "Dear customer your loan has been approved for $50,000. Click to get the money now.",
    "Free gift card offer limited time! Enter your details to receive your free Amazon card.",
    "You won the lottery! Send your bank details to claim $1 million prize immediately.",
    "Buy cheap medications online. Lowest prices guaranteed. No prescription required.",
    "Urgent action required your account will be deleted unless you verify your email now.",
    "Make money online guaranteed. Join thousands earning from home easily today.",
    "Exclusive offer just for you! Get 80% off on all products for limited time only.",
    "Your credit card has been charged $499. If this was not you call this number now.",
    "Win big at our online casino! Free spins no deposit required sign up now.",
    "Claim your free trial today no credit card needed. Cancel anytime guaranteed.",
    "Special discount for selected customers only. Buy now and save 70 percent.",
    "You have been pre-approved for a credit card with zero interest for life click here.",
    "ALERT your computer has a virus. Download our free antivirus software immediately.",
    "Investment opportunity earn 500 percent returns guaranteed call our experts now.",
    "Free iPhone giveaway click now to enter and win the latest Apple smartphone today.",
    "Nigerian prince needs your help to transfer millions reward you generously.",
    "You have unclaimed funds in your account verify identity now to release payment.",
    "Work from home earn dollars daily no skills required join our team now free.",
    "Online pharmacy lowest prices buy medications without prescription shipped fast.",
]

HAM_SAMPLES = [
    "Hey, are we still on for lunch tomorrow at noon? Let me know if anything changes.",
    "Please find the attached report for Q3 performance review. Let me know your feedback.",
    "Your package has been shipped and will arrive by Friday. Track your order online.",
    "Reminder: Your dentist appointment is scheduled for Monday at 2 PM.",
    "Hi Mom, just checking in. How are you feeling? Call me when you get a chance.",
    "The project meeting has been rescheduled to Thursday at 3 PM in conference room B.",
    "Thank you for your purchase. Your order #12345 has been confirmed.",
    "Can you review the pull request I submitted this morning? Waiting for your feedback.",
    "Your monthly bank statement is ready to view online in your account.",
    "Just a reminder that rent is due on the first of the month as per our agreement.",
    "Hey! I found a great recipe for the dish you wanted. I'll send it over tonight.",
    "The team lunch is tomorrow at the usual place. RSVP if you're coming please.",
    "Your flight to New York departs at 6:45 AM. Please arrive at airport two hours early.",
    "I reviewed your proposal and I think it looks great. Let's discuss it on Friday.",
    "The library book you requested is now available for pickup at the front desk.",
    "Happy birthday! Hope you have a wonderful day filled with joy and celebration.",
    "Please submit your timesheet by end of day Friday so payroll can be processed.",
    "The conference call is at 10 AM EST. Here is the dial-in information for tomorrow.",
    "Your subscription has been renewed automatically for another year thank you.",
    "Can we move our meeting to Wednesday? I have a conflict on Tuesday afternoon.",
    "The kids had a great time at soccer practice today. Coach said they are improving.",
    "Please review the contract and let me know if you have any changes or questions.",
    "Your test results are normal. No further action is needed at this time.",
    "The new software update is available. Please install it at your earliest convenience.",
    "I wanted to follow up on the proposal I sent last week. Any thoughts on it?",
    "Dinner was amazing last night. We should definitely go back to that restaurant again.",
    "Your account balance is $2,340.50 as of today. Have a great day.",
    "The homework assignment is due next Monday. Please submit it through the portal.",
    "We are excited to welcome you to our team starting next Monday morning.",
    "The weather forecast shows rain this weekend so the picnic might be postponed.",
]

def main():
    print("=" * 60)
    print("  AI Email Spam Detection System — Model Trainer")
    print("=" * 60)

    texts  = SPAM_SAMPLES + HAM_SAMPLES
    labels = ["spam"] * len(SPAM_SAMPLES) + ["ham"] * len(HAM_SAMPLES)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    detector = SpamDetector()
    print(f"\n[*] Training on {len(X_train)} emails...")
    detector.train(X_train, y_train)

    print("\n[*] Evaluating on test set...")
    results = detector.evaluate(X_test, y_test)
    print(f"\n  Accuracy : {results['accuracy']*100:.1f}%")
    print(f"\n{results['report']}")

    print("\n[*] Sample predictions:")
    samples = [
        ("Congratulations you won a million dollar prize click now!", "spam"),
        ("Hi, are you free for coffee tomorrow morning?", "ham"),
        ("URGENT: Claim your free gift before it expires today!", "spam"),
        ("Please review the attached quarterly report.", "ham"),
    ]
    for text, true_label in samples:
        result = detector.predict(text)
        status = "✓" if result['label'] == true_label else "✗"
        print(f"  [{status}] {text[:55]}...")
        print(f"       → {result['label'].upper()} ({result['confidence']*100:.1f}% confidence)\n")

    print("[✓] Training complete. Model saved to models/spam_model.pkl")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
