const state = {
    userId: null,
    username: "",
    chats: [],
    currentChatId: null,
};

const storageKeys = {
    userId: "chem_mas_user_id",
    username: "chem_mas_username",
};

let activeChatEventSource = null;
let liveProgressCard = null;
let toastTimer = null;

const elements = {
    loginOverlay: document.getElementById("login-overlay"),
    loginInput: document.getElementById("login-input"),
    loginBtn: document.getElementById("login-btn"),
    newChatBtn: document.getElementById("new-chat-btn"),
    chatList: document.getElementById("chat-list"),
    userName: document.getElementById("user-name"),
    logoutBtn: document.getElementById("logout-btn"),
    rateBtn: document.getElementById("rate-btn"),
    activeChatCaption: document.getElementById("active-chat-caption"),
    messagesContainer: document.getElementById("messages-container"),
    messageInput: document.getElementById("message-input"),
    sendBtn: document.getElementById("send-btn"),
    toast: document.getElementById("toast"),
};

async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
        },
        ...options,
    });

    if (!response.ok) {
        let detail = "Ошибка запроса";
        try {
            const payload = await response.json();
            detail = payload.detail || detail;
        } catch (_) {}
        throw new Error(detail);
    }

    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        return await response.json();
    }

    return response;
}

function saveSession(user) {
    localStorage.setItem(storageKeys.userId, user.user_id);
    localStorage.setItem(storageKeys.username, user.username);
}

function clearSession() {
    localStorage.removeItem(storageKeys.userId);
    localStorage.removeItem(storageKeys.username);
}

function applyUser(user) {
    state.userId = user.user_id;
    state.username = user.username;
    elements.userName.textContent = user.username;
    elements.loginOverlay.classList.add("hidden");
    elements.newChatBtn.disabled = false;
    elements.logoutBtn.disabled = false;
}

function resetAppState() {
    state.userId = null;
    state.username = "";
    state.chats = [];
    state.currentChatId = null;

    elements.userName.textContent = "Не выбран";
    elements.newChatBtn.disabled = true;
    elements.logoutBtn.disabled = true;
    elements.messageInput.disabled = true;
    elements.sendBtn.disabled = true;
    elements.messageInput.value = "";
    elements.activeChatCaption.textContent = "Выберите или создайте чат, чтобы начать работу";
    elements.chatList.innerHTML = "";
    renderEmptyState("Здесь появится переписка", "Создайте новый чат и отправьте запрос системе.");
    stopActiveChatEvents();
}

function showToast(text) {
    if (!text) {
        return;
    }

    elements.toast.textContent = text;
    elements.toast.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        elements.toast.classList.remove("visible");
    }, 3200);
}

function formatDate(value) {
    if (!value) {
        return "";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    return date.toLocaleString("ru-RU");
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderInlineMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return html;
}

function renderMarkdown(text) {
    const source = String(text || "").replace(/\r\n/g, "\n");
    const lines = source.split("\n");
    const blocks = [];

    let currentParagraph = [];
    let currentList = null;
    let inCodeFence = false;
    let codeFenceLines = [];

    const flushParagraph = () => {
        if (!currentParagraph.length) {
            return;
        }
        blocks.push(`<p>${currentParagraph.map((line) => renderInlineMarkdown(line)).join("<br>")}</p>`);
        currentParagraph = [];
    };

    const flushList = () => {
        if (!currentList || !currentList.items.length) {
            currentList = null;
            return;
        }

        const tag = currentList.type === "ordered" ? "ol" : "ul";
        const items = currentList.items
            .map((item) => `<li>${renderInlineMarkdown(item)}</li>`)
            .join("");
        blocks.push(`<${tag}>${items}</${tag}>`);
        currentList = null;
    };

    const flushCodeFence = () => {
        if (!inCodeFence) {
            return;
        }
        const code = escapeHtml(codeFenceLines.join("\n"));
        blocks.push(`<pre><code>${code}</code></pre>`);
        inCodeFence = false;
        codeFenceLines = [];
    };

    for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed.startsWith("```")) {
            flushParagraph();
            flushList();
            if (inCodeFence) {
                flushCodeFence();
            } else {
                inCodeFence = true;
                codeFenceLines = [];
            }
            continue;
        }

        if (inCodeFence) {
            codeFenceLines.push(line);
            continue;
        }

        const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
        if (headingMatch) {
            flushParagraph();
            flushList();
            const level = headingMatch[1].length;
            blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
            continue;
        }

        const orderedMatch = line.match(/^\s*\d+\.\s+(.+)$/);
        if (orderedMatch) {
            flushParagraph();
            if (!currentList || currentList.type !== "ordered") {
                flushList();
                currentList = { type: "ordered", items: [] };
            }
            currentList.items.push(orderedMatch[1]);
            continue;
        }

        const unorderedMatch = line.match(/^\s*[-*]\s+(.+)$/);
        if (unorderedMatch) {
            flushParagraph();
            if (!currentList || currentList.type !== "unordered") {
                flushList();
                currentList = { type: "unordered", items: [] };
            }
            currentList.items.push(unorderedMatch[1]);
            continue;
        }

        if (!trimmed) {
            flushParagraph();
            flushList();
            continue;
        }

        flushList();
        currentParagraph.push(line);
    }

    flushParagraph();
    flushList();
    flushCodeFence();

    return blocks.join("");
}

function updateComposerState() {
    const enabled = Boolean(state.userId && state.currentChatId);
    elements.messageInput.disabled = !enabled;
    elements.sendBtn.disabled = !enabled;
}

function renderEmptyState(title, text) {
    elements.messagesContainer.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-title">${escapeHtml(title)}</div>
            <div class="empty-state-text">${escapeHtml(text)}</div>
        </div>
    `;
}

function renderChatList() {
    elements.chatList.innerHTML = "";

    if (!state.chats.length) {
        const empty = document.createElement("div");
        empty.className = "chat-list-empty";
        empty.textContent = "Чатов пока нет";
        elements.chatList.appendChild(empty);
        return;
    }

    for (const chat of state.chats) {
        const item = document.createElement("div");
        item.className = "chat-item";
        if (chat.chat_id === state.currentChatId) {
            item.classList.add("active");
        }

        const main = document.createElement("button");
        main.type = "button";
        main.className = "chat-item-main";
        main.innerHTML = `
            <div class="chat-item-title">${escapeHtml(chat.title)}</div>
            <div class="chat-item-date">${formatDate(chat.updated_at)}</div>
        `;
        main.addEventListener("click", () => openChat(chat.chat_id));

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "chat-item-delete";
        remove.textContent = "×";
        remove.title = "Удалить чат";
        remove.addEventListener("click", (event) => {
            event.stopPropagation();
            deleteChat(chat.chat_id);
        });

        item.appendChild(main);
        item.appendChild(remove);
        elements.chatList.appendChild(item);
    }
}

function createMessageElement(message) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${message.role === "user" ? "message-user" : "message-assistant"}`;

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "ВЫ" : "АССИСТЕНТ";

    const body = document.createElement("div");
    body.className = "message-body";
    if (message.role === "assistant") {
        body.innerHTML = renderMarkdown(message.content);
    } else {
        body.textContent = message.content;
    }

    wrapper.appendChild(role);
    wrapper.appendChild(body);
    return wrapper;
}

function createProgressCard(messages = [], live = false) {
    const wrapper = document.createElement("div");
    wrapper.className = "progress-card";
    if (live) {
        wrapper.classList.add("live");
    }

    const title = document.createElement("div");
    title.className = "progress-title";
    title.textContent = "ХОД РАБОТЫ МУЛЬТИАГЕНТНОЙ СИСТЕМЫ";

    const subtitle = document.createElement("div");
    subtitle.className = "progress-subtitle";
    subtitle.textContent = "Главный агент координирует профильных агентов и собирает ответ";

    const list = document.createElement("div");
    list.className = "progress-list";

    wrapper.appendChild(title);
    wrapper.appendChild(subtitle);
    wrapper.appendChild(list);

    for (const message of messages) {
        appendProgressMessage(wrapper, message);
    }

    return wrapper;
}

function appendProgressMessage(wrapper, message) {
    const list = wrapper.querySelector(".progress-list");
    if (!list) {
        return;
    }

    const content = typeof message === "string" ? message : (message?.content || "");
    const agent = typeof message === "string" ? "" : (message?.agent || "");

    const line = document.createElement("div");
    line.className = "progress-line";

    if (agent) {
        const agentLabel = document.createElement("span");
        agentLabel.className = "progress-line-agent";
        agentLabel.textContent = `${agent}: `;
        line.appendChild(agentLabel);
    }

    const text = document.createElement("span");
    text.className = "progress-line-text";
    text.textContent = content;
    line.appendChild(text);

    list.appendChild(line);
    scrollMessagesToBottom();
}

function renderMessages(messages) {
    elements.messagesContainer.innerHTML = "";

    if (!messages.length) {
        renderEmptyState("В этом чате пока нет сообщений", "Отправьте первый запрос системе.");
        return;
    }

    let progressBuffer = [];

    const flushProgressBuffer = () => {
        if (!progressBuffer.length) {
            return;
        }
        elements.messagesContainer.appendChild(createProgressCard(progressBuffer, false));
        progressBuffer = [];
    };

    for (const message of messages) {
        if (message.role === "progress") {
            progressBuffer.push(message);
            continue;
        }

        flushProgressBuffer();
        elements.messagesContainer.appendChild(createMessageElement(message));
    }

    flushProgressBuffer();

    scrollMessagesToBottom();
}

function scrollMessagesToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

async function login() {
    const username = elements.loginInput.value.trim();
    if (!username) {
        showToast("Введите логин.");
        return;
    }

    elements.loginBtn.disabled = true;
    try {
        const user = await apiRequest("/api/session/login", {
            method: "POST",
            body: JSON.stringify({ username }),
        });
        saveSession(user);
        applyUser(user);
        await loadChats({ autoOpenFirst: true });
        if (!state.chats.length) {
            renderEmptyState("Чатов пока нет", "Нажмите «+ Новый чат», чтобы начать работу.");
        }
    } catch (error) {
        showToast(error.message);
    } finally {
        elements.loginBtn.disabled = false;
    }
}

async function restoreSession() {
    const userId = localStorage.getItem(storageKeys.userId);
    if (!userId) {
        resetAppState();
        elements.loginOverlay.classList.remove("hidden");
        return;
    }

    try {
        const user = await apiRequest(`/api/users/${userId}`);
        applyUser(user);
        await loadChats({ autoOpenFirst: true });
        if (!state.chats.length) {
            renderEmptyState("Чатов пока нет", "Нажмите «+ Новый чат», чтобы начать работу.");
        }
    } catch (_) {
        clearSession();
        resetAppState();
        elements.loginOverlay.classList.remove("hidden");
    }
}

async function loadChats({ autoOpenFirst = false } = {}) {
    if (!state.userId) {
        return;
    }

    const data = await apiRequest(`/api/users/${state.userId}/chats`);
    state.chats = data.items || [];
    renderChatList();

    if (state.currentChatId) {
        const stillExists = state.chats.some((chat) => chat.chat_id === state.currentChatId);
        if (!stillExists) {
            state.currentChatId = null;
        }
    }

    if (!state.currentChatId && autoOpenFirst && state.chats.length) {
        await openChat(state.chats[0].chat_id);
    } else {
        updateComposerState();
    }
}

async function createNewChat() {
    if (!state.userId) {
        showToast("Сначала введите логин.");
        return;
    }

    const data = await apiRequest(`/api/users/${state.userId}/chats`, {
        method: "POST",
        body: JSON.stringify({}),
    });

    await loadChats();
    await openChat(data.chat_id);
}

async function openChat(chatId) {
    if (!state.userId) {
        return;
    }

    stopActiveChatEvents();
    const data = await apiRequest(`/api/users/${state.userId}/chats/${chatId}`);
    state.currentChatId = data.chat_id;
    renderChatList();
    renderMessages(data.messages || []);
    elements.activeChatCaption.textContent = data.title || "Новый чат";
    updateComposerState();
    elements.messageInput.focus();
}

async function deleteChat(chatId) {
    if (!state.userId) {
        return;
    }

    await apiRequest(`/api/users/${state.userId}/chats/${chatId}`, {
        method: "DELETE",
    });

    if (state.currentChatId === chatId) {
        state.currentChatId = null;
        updateComposerState();
    }

    await loadChats({ autoOpenFirst: true });
    if (!state.currentChatId) {
        elements.activeChatCaption.textContent = "Выберите или создайте чат, чтобы начать работу";
        renderEmptyState("Чат удалён", "Создайте новый чат или откройте существующий.");
    }
}

function stopActiveChatEvents({ clearProgress = true } = {}) {
    if (activeChatEventSource) {
        activeChatEventSource.close();
        activeChatEventSource = null;
    }
    if (clearProgress) {
        liveProgressCard = null;
    }
}

function startChatEventStream(chatId) {
    stopActiveChatEvents({ clearProgress: false });

    const eventSource = new EventSource(`/api/users/${state.userId}/chats/${chatId}/events`);
    activeChatEventSource = eventSource;

    eventSource.addEventListener("progress", (event) => {
        if (state.currentChatId !== chatId) {
            return;
        }

        try {
            const payload = JSON.parse(event.data);
            if (!liveProgressCard) {
                liveProgressCard = createProgressCard([], true);
                elements.messagesContainer.appendChild(liveProgressCard);
            }
            appendProgressMessage(liveProgressCard, payload);
        } catch (_) {}
    });

    eventSource.addEventListener("completed", () => {
        stopActiveChatEvents();
    });

    eventSource.addEventListener("error", (event) => {
        if (state.currentChatId !== chatId) {
            return;
        }
        try {
            const payload = JSON.parse(event.data);
            if (!liveProgressCard) {
                liveProgressCard = createProgressCard([], true);
                elements.messagesContainer.appendChild(liveProgressCard);
            }
            appendProgressMessage(liveProgressCard, payload);
        } catch (_) {}
    });

    eventSource.onerror = () => {
        if (activeChatEventSource === eventSource) {
            eventSource.close();
            activeChatEventSource = null;
        }
    };
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();

    if (!state.currentChatId) {
        showToast("Сначала создайте или выберите чат.");
        return;
    }

    if (!message) {
        return;
    }

    if (message.length > 500) {
        showToast("Сообщение должно быть не длиннее 500 символов.");
        return;
    }

    elements.sendBtn.disabled = true;
    elements.messageInput.disabled = true;

    elements.messagesContainer.appendChild(
        createMessageElement({ role: "user", content: message })
    );
    liveProgressCard = createProgressCard([], true);
    elements.messagesContainer.appendChild(liveProgressCard);
    scrollMessagesToBottom();

    elements.messageInput.value = "";
    startChatEventStream(state.currentChatId);

    try {
        const result = await apiRequest(`/api/users/${state.userId}/chats/${state.currentChatId}/messages`, {
            method: "POST",
            body: JSON.stringify({ message }),
        });

        stopActiveChatEvents();
        await openChat(state.currentChatId);
        await loadChats();

        if (!result.success && result.error) {
            showToast(result.error);
        }
    } catch (error) {
        stopActiveChatEvents();
        showToast(error.message);
        await openChat(state.currentChatId);
    } finally {
        updateComposerState();
    }
}

function logout() {
    clearSession();
    resetAppState();
    elements.loginInput.value = "";
    elements.loginOverlay.classList.remove("hidden");
    elements.loginInput.focus();
}

function handleRate() {
    showToast("Окно оценки можно подключить следующим шагом.");
}

elements.loginBtn.addEventListener("click", login);
elements.newChatBtn.addEventListener("click", createNewChat);
elements.sendBtn.addEventListener("click", sendMessage);
elements.logoutBtn.addEventListener("click", logout);
elements.rateBtn.addEventListener("click", handleRate);

elements.loginInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        login();
    }
});

elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

window.addEventListener("load", async () => {
    resetAppState();
    await restoreSession();
});
