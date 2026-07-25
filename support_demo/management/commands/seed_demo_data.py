"""
Seed support demo data for the Academy demo.

    python manage.py seed_demo_data
"""
from django.core.management.base import BaseCommand

from support_demo.models import SupportTicket

DEMO_TICKETS = [
    {
        "title": "Login page returns 500 error after update",
        "body": (
            "Since the latest deployment, users are getting a 500 Internal Server Error "
            "when they try to log in. The error started at around 14:00 UTC yesterday. "
            "We have 500+ affected users. This is blocking production."
        ),
    },
    {
        "title": "How do I export my data to CSV?",
        "body": (
            "I would like to export all my records to a CSV file for analysis in Excel. "
            "I cannot find this option anywhere in the settings. Is it available? "
            "If not, can it be added as a feature?"
        ),
    },
    {
        "title": "Payment declined but money was charged",
        "body": (
            "I tried to purchase the Pro plan but the payment failed. "
            "However, I can see the charge of £49.99 on my credit card statement. "
            "I have not received any confirmation email. Please refund or activate my account."
        ),
    },
    {
        "title": "Dashboard loads very slowly",
        "body": (
            "The main dashboard takes over 30 seconds to load. "
            "This started happening about a week ago. Before that it was fast. "
            "I have tried on different browsers and internet connections — same result."
        ),
    },
]


class Command(BaseCommand):
    help = "Seed support demo tickets and run the full seed pipeline."

    def add_arguments(self, parser):
        parser.add_argument(
            "--with-docs",
            action="store_true",
            help="Also import platform and Academy documentation.",
        )

    def handle(self, *args, **options):
        self.stdout.write("Seeding demo data...")

        # Run the training data seed first
        from django.core.management import call_command
        call_command("seed_academy_training_data")

        # Import docs if requested
        if options["with_docs"]:
            self.stdout.write("Importing documentation...")
            call_command("import_academy_docs")

        # Create demo tickets
        created = 0
        for ticket_data in DEMO_TICKETS:
            _, was_created = SupportTicket.objects.get_or_create(
                title=ticket_data["title"],
                defaults={"body": ticket_data["body"]},
            )
            if was_created:
                created += 1
                self.stdout.write(f"  Created ticket: {ticket_data['title'][:60]}")

        self.stdout.write(self.style.SUCCESS(
            f"Done. {created} demo tickets created. "
            "Open /admin/ to explore. Use 'Run AI triage' on tickets to test the workflow."
        ))
