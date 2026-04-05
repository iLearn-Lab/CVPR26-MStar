import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

def send_error_email(error_msg, subject="Python Script Error Notification"):
    smtp_server = ""
    smtp_port = 465
    sender_email = ""
    sender_password = ""
    receiver_email = ""

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    body = f"""
    An error occurred in your Python script:
    {error_msg}

    Script: {sys.argv[0]}
    """
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Error notification email sent successfully.")
    except Exception as e:
        print(f"Failed to send error email: {str(e)}")
