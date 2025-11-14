# Flowbotics AI Integration Guides

## Platform Integration Overview

This guide provides step-by-step instructions for integrating Flowbotics AI chatbots across different platforms and third-party services.

---

## 1. WEBSITE INTEGRATION

### HTML/JavaScript Integration

#### Step 1: Get Your Chatbot ID
- Log in to Flowbotics Dashboard
- Navigate to "Deployments" → "Website"
- Copy your unique Chatbot ID

#### Step 2: Add Embed Code to Your Website

**Option A: Add to HTML Footer**

```html
<!-- Add this before closing </body> tag -->
<script src="https://cdn.flowbotics.ai/chatbot.min.js"></script>
<script>
  Flowbotics.init({
    chatbotId: 'YOUR_CHATBOT_ID',
    position: 'bottom-right', // bottom-right, bottom-left, top-right, top-left
    theme: 'light', // light, dark, custom
    primaryColor: '#2563EB', // Your brand color
    welcomeMessage: 'Hi! How can we help you?'
  });
</script>
```

**Option B: Using NPM (For React/Vue/Angular)**

```bash
npm install @flowbotics/chatbot-sdk
```

```javascript
import { FlowboticsChat } from '@flowbotics/chatbot-sdk';

<FlowboticsChat
  chatbotId="YOUR_CHATBOT_ID"
  position="bottom-right"
  theme="light"
  primaryColor="#2563EB"
/>
```

#### Step 3: Customize Appearance

```javascript
Flowbotics.init({
  chatbotId: 'YOUR_CHATBOT_ID',
  
  // Positioning
  position: 'bottom-right',
  marginBottom: '20px',
  marginRight: '20px',
  
  // Styling
  theme: 'custom',
  primaryColor: '#2563EB',
  secondaryColor: '#F3F4F6',
  textColor: '#1F2937',
  
  // Messages
  welcomeMessage: 'Welcome to Flowbotics AI!',
  headerTitle: 'Flowbotics Support',
  placeholderText: 'Type your question...',
  
  // Behavior
  minimized: false, // Start minimized
  animateButton: true,
  soundEnabled: true,
  
  // Features
  showRating: true,
  showFeedback: true,
  allowAttachments: true
});
```

#### Step 4: Track Events (Optional)

```javascript
// Track when user opens chat
Flowbotics.on('chatOpened', function() {
  console.log('Chat opened');
  // Send analytics
  gtag('event', 'flowbotics_chat_open');
});

// Track messages
Flowbotics.on('messageSent', function(message) {
  console.log('Message sent:', message);
});

// Track feedback
Flowbotics.on('feedbackSubmitted', function(rating) {
  console.log('Feedback rating:', rating);
});
```

---

## 2. WHATSAPP INTEGRATION

### Prerequisites
- WhatsApp Business Account
- WhatsApp Business API Access (via Meta)
- Phone number verified with WhatsApp

### Setup Steps

#### Step 1: Connect WhatsApp Business Account

1. Go to Flowbotics Dashboard → Integrations
2. Click "Connect WhatsApp"
3. Log in with your Meta/Facebook account
4. Select your WhatsApp Business Account
5. Authorize Flowbotics AI

#### Step 2: Configure Webhook

1. In Flowbotics Dashboard → WhatsApp Settings
2. Copy your Webhook URL
3. Go to Meta App Dashboard → WhatsApp → Configuration
4. Enter webhook URL and verify token

#### Step 3: Set Welcome Message

```
Settings → WhatsApp → Welcome Message

"Hi there! 👋 Welcome to Flowbotics AI. 

How can I help you today?
1️⃣ Chat with our assistant
2️⃣ Check order status
3️⃣ Speak to support team

Reply with your choice number!"
```

#### Step 4: Configure Auto-Responses

```
Auto Response Settings:
- Enable: Yes
- Delay: 2 seconds
- Message: "Thanks for reaching out! Our AI assistant is here to help. 🤖"
```

#### Step 5: Message Templates

Create pre-approved templates for better delivery:

```
Template: order_status
Language: English
Content:

Hi {{customer_name}},

Your order #{{order_id}} status:
📦 Status: {{status}}
📅 Expected: {{delivery_date}}

Need help? Reply 'HELP'
```

### WhatsApp Features Available

| Feature | Available |
|---------|-----------|
| Text messages | ✅ |
| Images | ✅ |
| Documents | ✅ |
| Quick replies | ✅ |
| Buttons | ✅ |
| Templates | ✅ |
| Media | ✅ |
| Location sharing | ✅ |
| Voice notes | ✅ |
| Video calls | ✅ (link only) |

---

## 3. INSTAGRAM INTEGRATION

### Prerequisites
- Instagram Business Account
- Meta Business Account
- Flowbotics AI Professional+ Plan

### Setup Steps

#### Step 1: Connect Instagram Account

1. Flowbotics Dashboard → Integrations
2. Click "Connect Instagram"
3. Log in with Facebook/Meta account
4. Select Instagram Business Account
5. Grant permissions

#### Step 2: Enable Direct Messages

1. Navigate to Instagram Settings
2. Ensure DMs from business are enabled
3. Set up automated responses

#### Step 3: Configure Bot Behavior

```
Instagram Bot Settings:
- Auto-reply: Enabled
- Auto-reply delay: 3 seconds
- Response type: Chatbot first, then escalate to human
- Escalation timeout: 5 minutes
```

#### Step 4: Message Rules

```
Instagram Message Rules:

Rule 1: Greeting
Trigger: First message from new follower
Response: "Hi {{user.first_name}}! 👋 Thanks for reaching out. How can Flowbotics help?"

Rule 2: FAQ
Trigger: Messages containing "price", "cost", "how much"
Response: Fetch pricing from knowledge base

Rule 3: Orders
Trigger: Messages containing "order", "track", "status"
Response: Connect to order tracking system
```

### Instagram Features

| Feature | Available |
|---------|-----------|
| Direct messages | ✅ |
| Instagram Stories replies | ✅ |
| Quick replies | ✅ |
| Carousel posts | ✅ |
| Hashtag responses | ⚠️ (Limited) |
| Comments | ⚠️ (Limited) |

---

## 4. CRM INTEGRATION (SALESFORCE)

### Prerequisites
- Salesforce account (any edition)
- Salesforce Connected App setup
- API permissions enabled

### Step-by-Step Setup

#### Step 1: Create Salesforce Connected App

1. Log in to Salesforce
2. Go to Setup → Apps → App Manager
3. Click "New Connected App"
4. Fill in details:
   - Connected App Name: "Flowbotics AI"
   - API Name: "Flowbotics_AI"
   - Contact Email: your@email.com

#### Step 2: Enable OAuth

1. Under "API (Enable OAuth Settings)"
2. Check "Enable OAuth Settings"
3. Add Callback URL: `https://api.flowbotics.ai/oauth/salesforce/callback`
4. Select OAuth Scopes:
   - Access and manage your data (api)
   - Full access until you revoke it (full, refresh_token, offline_access)

#### Step 3: Get Credentials

1. Save the Connected App
2. Click "View" on your app
3. Copy:
   - Consumer Key
   - Consumer Secret
   - Login URL

#### Step 4: Connect in Flowbotics

1. Flowbotics Dashboard → Integrations → CRM
2. Select "Salesforce"
3. Enter:
   - Consumer Key
   - Consumer Secret
   - Salesforce Instance URL
4. Click "Authorize"

#### Step 5: Configure Data Sync

```
Sync Configuration:

Lead Sync:
- Sync when: New chat starts with email
- Create field mapping:
  - Chat.user_name → Lead.FirstName + LastName
  - Chat.email → Lead.Email
  - Chat.company → Lead.Company
  - Chat.phone → Lead.Phone

Update if exists: Yes
Default Lead Source: "Flowbotics AI Chat"
Default Status: "New"

Contact Sync:
- Sync when: Existing customer identified
- Fields: Same as above

Opportunity Sync:
- Create opportunity when: Chat indicates purchase intent
- Mapping:
  - Chatbot product recommendation → Opportunity.Name
  - Chat value estimate → Opportunity.Amount
```

#### Step 6: Test Integration

```
Test Steps:
1. Start a test chat
2. Provide name, email, company
3. Go to Salesforce → Leads
4. Look for new lead created
5. Verify all fields populated correctly
```

---

## 5. PAYMENT GATEWAY INTEGRATION (STRIPE)

### Prerequisites
- Stripe account
- Stripe API keys
- Webhook access enabled

### Setup Steps

#### Step 1: Get Stripe API Keys

1. Log in to Stripe Dashboard
2. Navigate to Developers → API Keys
3. Copy:
   - Publishable Key
   - Secret Key

#### Step 2: Configure in Flowbotics

1. Dashboard → Integrations → Payments
2. Select "Stripe"
3. Paste API keys
4. Enable webhooks

#### Step 3: Set Up Webhook

1. Stripe Dashboard → Webhooks
2. Add endpoint: `https://api.flowbotics.ai/webhooks/stripe`
3. Select events:
   - `payment_intent.succeeded`
   - `payment_intent.payment_failed`
   - `charge.refunded`
   - `charge.dispute.created`

#### Step 4: Create Payment Flow in Chatbot

```
Chatbot Conversation Flow:

Bot: "Would you like to proceed with payment?"
User: "Yes"
Bot: [Shows payment button/link]
User: [Completes Stripe payment]
Webhook: [Triggers confirmation]
Bot: "Payment successful! Order #12345 confirmed."
Salesforce: [Creates opportunity/deal]
Google Sheets: [Logs transaction]
```

#### Step 5: Test Payment

```
Use Stripe test card: 4242 4242 4242 4242
Expiry: Any future date
CVC: Any 3 digits

Expected result: 
- Payment processes
- Flowbotics receives confirmation
- Chat confirms success
```

---

## 6. GOOGLE SHEETS AUTOMATION

### Prerequisites
- Google Account
- Google Sheets document
- Flowbotics Professional+ plan

### Setup Steps

#### Step 1: Create Google Sheet

Create a sheet with columns:
```
A: Timestamp
B: User Name
C: Email
D: Message
E: Chat Session ID
F: Response
G: Rating
H: Feedback
```

#### Step 2: Share Sheet with Flowbotics

1. Click "Share" on your sheet
2. Get the sheet URL
3. In Flowbotics Dashboard → Integrations → Google Sheets
4. Paste sheet URL
5. Click "Authorize"

#### Step 3: Configure Data Mapping

```
Column Mapping:
- Timestamp → A (Auto)
- User.name → B
- User.email → C
- Chat.message → D
- Chat.session_id → E
- Bot.response → F
- User.rating → G
- User.feedback → H
```

#### Step 4: Set Sync Frequency

```
Sync Settings:
- Trigger: Real-time
- Add row when: Message sent
- Backup: Daily
- Keep history: 90 days
```

#### Step 5: Create Formulas

```
In Column I (Sentiment Score):
=IF(OR(G:G>4, SEARCH("great", H:H)), "Positive", 
   IF(OR(G:G<3, SEARCH("bad", H:H)), "Negative", "Neutral"))

In Column J (Escalation Flag):
=IF(OR(G:G=1, G:G=2, SEARCH("escalate", H:H)), "YES", "NO")

Create pivot table to analyze:
- Ratings distribution
- Common feedback themes
- Escalation rate
```

---

## 7. EMAIL AUTOMATION (GMAIL)

### Prerequisites
- Google Workspace account
- Gmail enabled
- Flowbotics Professional+ plan

### Setup Steps

#### Step 1: Configure Gmail Integration

1. Dashboard → Integrations → Email
2. Select "Gmail"
3. Click "Connect Google Account"
4. Grant permissions to:
   - Read emails
   - Send emails
   - Manage labels
   - Manage filters

#### Step 2: Set Up Email Rules

```
Rule 1: Auto-Respond to Sales Inquiries
- Trigger: Email contains "sales", "pricing", "quote"
- Action: Send auto-response from template
- Template: "Thanks for inquiry! Our team will contact you within 2 hours"
- Add to label: "Sales-AutoReply"

Rule 2: Classify Support Tickets
- Trigger: Email from support email
- Action: Extract info and create in Salesforce
- Add to label: "Support-Tickets"

Rule 3: Forward Urgent Issues
- Trigger: Email contains "urgent", "critical", "ASAP"
- Action: Forward to team email
- Label: "Urgent"
```

#### Step 3: Create Email Templates

```
Template: Sales Inquiry Response
Subject: Re: {{subject}}

Hi {{name}},

Thanks for your interest in Flowbotics AI!

Our {{product}} package offers:
✓ Multi-channel support
✓ 24/7 automated responses
✓ Real-time analytics

I'd love to discuss your needs. Are you available for a brief call this week?

Best regards,
Flowbotics Team
---
[Link to calendar]
```

#### Step 4: Enable Auto-Categorization

```
Use AI to automatically:
- Extract sender email & name
- Identify inquiry type (sales, support, feedback)
- Extract key information
- Route to correct handler
- Send appropriate response
```

---

## 8. MICROSOFT TEAMS INTEGRATION

### Prerequisites
- Microsoft Teams account
- Flowbotics Professional+ plan
- Teams app permissions

### Setup Steps

#### Step 1: Register Teams App

1. Go to [Microsoft App Registration Portal](https://portal.azure.com/)
2. Register new application
3. Note App ID and Tenant ID
4. Create app password

#### Step 2: Connect to Flowbotics

1. Dashboard → Integrations → Microsoft Teams
2. Enter:
   - App ID
   - App Password
   - Tenant ID
3. Click "Connect"

#### Step 3: Deploy to Teams

1. Download Flowbotics manifest.json
2. Go to Teams → Apps → Create app → Import custom app
3. Upload manifest.json
4. Install to your workspace

#### Step 4: Configure Bot Commands

```
Commands:
- /help → Show available features
- /faq → Common questions
- /chat → Start conversation
- /feedback → Rate last response
- /escalate → Talk to human agent
```

---

## 9. SLACK INTEGRATION

### Prerequisites
- Slack workspace
- Slack admin access
- Flowbotics Professional+ plan

### Setup Steps

#### Step 1: Create Slack App

1. Go to [Slack App Directory](https://api.slack.com/apps)
2. Create New App
3. Name: "Flowbotics AI Chatbot"
4. Select workspace

#### Step 2: Configure Permissions

Required scopes:
```
- chat:write
- chat:write.public
- app_mentions:read
- message_mentions:read
- commands
- incoming-webhooks
```

#### Step 3: Set Up Event Subscriptions

1. Enable Events
2. Request URL: `https://api.flowbotics.ai/slack/events`
3. Subscribe to bot events:
   - app_mention
   - message.im
   - message.channels

#### Step 4: Configure Commands

```
/ask [question] → Ask Flowbotics AI
/feedback [rating] [comment] → Rate responses
/help → Show available commands
/status → Get system status
```

#### Step 5: Install to Workspace

1. Under "Install App"
2. Click "Install to Workspace"
3. Authorize permissions

---

## 10. CUSTOM API INTEGRATION

### Authentication

```javascript
// OAuth 2.0 Bearer Token
const token = 'your_api_token_here';
const headers = {
  'Authorization': `Bearer ${token}`,
  'Content-Type': 'application/json'
};
```

### Send Message

```javascript
fetch('https://api.flowbotics.ai/v1/chat', {
  method: 'POST',
  headers: headers,
  body: JSON.stringify({
    message: 'What services do you offer?',
    session_id: 'user_session_123',
    user_id: 'user_456',
    metadata: {
      source: 'custom_app',
      context: 'product_inquiry'
    }
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### Response Format

```json
{
  "status": "success",
  "data": {
    "response": "Flowbotics AI offers AI chatbots and business process automation...",
    "session_id": "user_session_123",
    "confidence_score": 0.92,
    "source_documents": ["services.md", "pricing.md"],
    "suggested_follow_ups": [
      "What's the pricing?",
      "How long does implementation take?"
    ]
  }
}
```

---

## Troubleshooting Guide

### Common Issues

**Issue: Chatbot not responding**
- ✓ Check API keys
- ✓ Verify webhook URL is accessible
- ✓ Check rate limits
- ✓ Review error logs in dashboard

**Issue: Data not syncing to CRM**
- ✓ Verify API permissions
- ✓ Check field mapping
- ✓ Ensure lead/contact records exist
- ✓ Review sync logs

**Issue: WhatsApp messages delayed**
- ✓ Check WhatsApp template approval
- ✓ Verify phone number is active
- ✓ Check message character limit
- ✓ Review rate limits

**Issue: Stripe payment fails**
- ✓ Verify API keys are current
- ✓ Check webhook delivery
- ✓ Ensure card is valid
- ✓ Review Stripe logs

### Support

For integration help:
- Email: integrations@flowbotics.ai
- Chat: In-app support
- Docs: [docs.flowbotics.ai](https://docs.flowbotics.ai)

