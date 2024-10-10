function toggleMenu(){
    const menu=document.querySelector(".menu-links");
    const icon=document.querySelector(".hamburger-icon");
    menu.classList.toggle("open");
    icon.classList.toggle("open");
}

// For IBM Watson Integration //
window.watsonAssistantChatOptions = {
    integrationID: "bf4b8cc6-95c5-427c-8aa2-d2897483bb41", // The ID of this integration.
    region: "au-syd", // The region your integration is hosted in.
    serviceInstanceID: "315197c8-8461-4857-bf2e-7e6f99c1a270", // The ID of your service instance.
    onLoad: function(instance) { instance.render(); }
  };
  setTimeout(function(){
    const t = document.createElement('script');
    t.src = "https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
    document.head.appendChild(t);
  });